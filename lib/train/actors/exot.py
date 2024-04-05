from re import L
from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search


class EXOTActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""
    def __init__(self, net, objective, loss_weight, settings, loss_type):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.exit_flag = loss_type

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict, flagFeat = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)
        template_bboxes = box_xywh_to_xyxy(data['template_anno']) 
        gt_package = [template_bboxes]

        gt_exit = torch.squeeze(data['search_exit'])

        # compute losses
        if gt_bboxes.dim() ==3 and gt_bboxes.shape[0]==1:
            gt_bboxes = gt_bboxes[0]

        loss, status = self.compute_losses(out_dict, gt_bboxes, flag_feat = flagFeat)
        
        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        feat_dict_list = []
        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            feat_dict_list.append(self.net(img=NestedTensor(template_img_i, template_att_i), mode='backbone'))

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        feat_dict_list.append(self.net(img=NestedTensor(search_img, search_att), mode='backbone'))

        # run the transformer and compute losses
        # for odin classification, return_search = True
        seq_dict = merge_template_search(feat_dict_list, return_search=True, return_template=True)
        
        template_bboxes = box_xywh_to_xyxy(data['template_anno'])  #(N_t, batch, 4)

        # search_joint = data['search_joint'] #(N_s, batch, 6)
        # print('joint flag', data['joint_flag'])
        # if data['joint_flag'][0] != 'None':
        #     template_joint = data['template_joint'] #(N_t, batch, 6)
        # else:
        template_joint = None
        joint_annot = (template_bboxes, template_joint)  # template anno, template joint
        out_dict, _, _, flagFeat = self.net(seq_dict=seq_dict, annot = joint_annot, mode="transformer", run_box_head=run_box_head, run_cls_head=run_cls_head)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict, flagFeat

    def compute_losses(self, pred_dict, gt_bbox, flag_feat = None, return_status=True):
        # if len(flag_feat) == 3:
        #     gt_exit, gt_package, epoch = flag_feat
        #     exitflag = None; feature = None
        # else:
        #     gt_exit, gt_package, feature, exitflag, epoch = flag_feat
        
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)

        pred_bboxes_vec = box_cxcywh_to_xyxy(pred_boxes)
        pred_boxes_vec = pred_bboxes_vec.view(-1, 4) # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        if gt_bbox.dim() ==3:
            gt_bboxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(min=-1.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            gt_boxes_vec = gt_bboxes_vec.view(-1,4).clamp(min=0.0, max=1.0)
            
        else:
            tmp = box_xywh_to_xyxy(gt_bbox)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :]
            gt_bboxes_vec = gt_boxes_vec.repeat((1, num_queries, 1)).clamp(min=-1.0, max=1.0)
            gt_boxes_vec = gt_bboxes_vec.view(-1, 4).clamp(min=0.0, max=1.0)              # (B,4) --> (B,1,4) --> (B,N,4)
            
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        
        # if self.exit_flag == 'None':
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss

        
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()

            status = {"Loss/total": loss.item(),
                          "Loss/giou": giou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "IoU": mean_iou.item(),
                        "total_loss": loss.item(),

                          }
            return loss, status
        else:
            return loss

    def compute_joint_loss(self, gt_bbox, pred_dict, gt_package):
        if len(gt_package) >1:
            # print(gt_package[0].shape, gt_package[1].shape, gt_package[2].shape)
            # torch.Size([3, 6]) torch.Size([3, 480, 640]) torch.Size([2, 3, 480, 640])
            [gt_joints, gt_depth, template_depth, template_bboxes] = gt_package
            if gt_bbox.dim()==3:
                gt_joints = gt_joints[:,:,:2].view(-1, 2)
            else:
                gt_joints = gt_joints[:, :2]
            pred_joints = pred_dict['pred_joint']
            # print(gt_joints.shape, pred_joints.shape)
            if pred_joints !=None:
                joint_loss = self.objective['joint'](gt_joints, pred_joints)
                l1_loss = l1_loss + joint_loss/2
            else:
                joint_loss = torch.tensor(0.0)
            template_bboxes = template_bboxes.clamp(min=-1.0, max=1.0) 
            #reid_loss += self.compute_reid_depth(pred_bboxes_vec, template_bboxes, gt_depth, template_depth)
            #reid_loss = reid_loss/2
        else:
            joint_loss = torch.tensor(0.0)
        return joint_loss, l1_loss

    def compute_exit_loss(self, exitflag, neg_flags, gt_bboxes_vec, gt_exit):
        if exitflag == None:
            exit_loss = torch.tensor(0.0).cuda()
            return exit_loss
        if self.exit_flag == "BCE":
            exit_flag12 = torch.squeeze(exitflag[-1][2])
            exit_flag13 = torch.squeeze(exitflag[-1][3])
            exit_loss = (self.objective['exit_top'](exit_flag12, gt_exit) + self.objective['exit_top'](exit_flag13, gt_exit))/2
            #exit_loss = self.objective['exit_bottom'](neg_flags, exitflag[-1][0]) + self.objective['exit_bottom'](neg_flags, exitflag[-1][1])
        elif self.exit_flag == 'MATRIX_BCE':
            exit_loss = self.cal_exit_prob(gt_bboxes_vec, exitflag[0], exitflag[1])
        elif self.exit_flag == 'LAMBDA':

            tmppos_tl = exitflag[-1][0][neg_flags==0]
            tmppos_br = exitflag[-1][1][neg_flags==0]
            if tmppos_tl.shape[0] ==0:
                exit_loss = torch.tensor(0.0).cuda()
            else:
                exit_loss = torch.mean(tmppos_tl)
            if tmppos_br.shape[0] ==0:
                exit_loss += torch.tensor(0.0).cuda()
            else:
                exit_loss += torch.mean(tmppos_br)

            tmpneg_tl = exitflag[-1][0][neg_flags==1] 
            tmpneg_br = exitflag[-1][1][neg_flags==1]
            if tmpneg_tl.shape[0] ==0:
                exit_loss -= torch.tensor(0.0).cuda()
            else:
                exit_loss -= torch.mean(tmpneg_tl)
            if tmpneg_br.shape[0] ==0:
                exit_loss -= torch.tensor(0.0).cuda()
            else:
                exit_loss -= torch.mean(tmpneg_br)
        else:
            raise Exception('Invalid exit loss')
        return exit_loss

    def make_gt_matrix(self, feat_sz, tgt_idx):
        gt_matrix = torch.zeros(feat_sz*feat_sz).view(feat_sz, feat_sz).cuda()
        for i in range(feat_sz):
            for j in range(feat_sz):
                gt_matrix[i][j] = torch.exp(-(j-tgt_idx[0])**2-(i-tgt_idx[1])**2)
        gt_matrix = torch.squeeze(gt_matrix.view(-1, feat_sz*feat_sz))
        return gt_matrix

    def make_uni_matrix(self, feat_sz):
        gt_matrix = torch.ones(feat_sz*feat_sz).cuda()/(feat_sz*feat_sz)

        return gt_matrix
    
    def cal_exit_prob(self, gt_bboxes, prob_vec_tl, prob_vec_br):
        feat_sz = 20
        gt_bboxes = torch.squeeze(gt_bboxes)
        index = torch.round(gt_bboxes*feat_sz).int()
        ent_matrix_loss = 0
        neg_tup = torch.nonzero(gt_bboxes<0, as_tuple=True)
        neg_n = torch.unique(neg_tup[0])
        neg_b = torch.unique(neg_tup[1])

        if gt_bboxes.dim()==3:
            n, b, _ = gt_bboxes.shape
            for i in range(n):
                for j in range(b):
                    if i in neg_n and j in neg_b:
                        gt_matrix = self.make_uni_matrix(feat_sz)
                        ent_matrix_loss += self.objective['exit_top'](prob_vec_tl[i,j], gt_matrix)
                        # print("GT UNI matrix", gt_matrix)
                        gt_matrix = self.make_uni_matrix(feat_sz)
                        ent_matrix_loss += self.objective['exit_top'](prob_vec_br[i,j], gt_matrix)
                    else:
                        gt_matrix = self.make_gt_matrix(feat_sz, [index[i,j,0], index[i,j,1]])
                        ent_matrix_loss += self.objective['exit_top'](prob_vec_tl[i,j], gt_matrix)
                        # print("TL gaussian matrix", gt_matrix)
                        gt_matrix = self.make_gt_matrix(feat_sz, [index[i,j,2], index[i,j,3]])
                        ent_matrix_loss += self.objective['exit_top'](prob_vec_br[i,j], gt_matrix)
            ent_matrix_loss = ent_matrix_loss/(n*b)
        else:
            b, _ = gt_bboxes.shape

            for j in range(b):
                if j in neg_n:
                    gt_matrix = self.make_uni_matrix(feat_sz)
                    ent_matrix_loss += self.objective['exit_top'](prob_vec_tl[j], gt_matrix)
                    gt_matrix = self.make_uni_matrix(feat_sz)
                    ent_matrix_loss += self.objective['exit_top'](prob_vec_br[j], gt_matrix)
                else:
                    gt_matrix = self.make_gt_matrix(feat_sz, [index[j,0], index[j,1]])
                    ent_matrix_loss += self.objective['exit_top'](prob_vec_tl[j], gt_matrix)
                    gt_matrix = self.make_gt_matrix(feat_sz, [index[j,2], index[j,3]])
                    ent_matrix_loss += self.objective['exit_top'](prob_vec_br[j], gt_matrix)
            ent_matrix_loss = ent_matrix_loss/(b)
        
        return ent_matrix_loss

    def compute_reid_loss(self, gt_bbox, pred_bboxes_vec, feature):
        #compute re-id loss
        if feature == None:
            reid_loss = torch.tensor(0.0).cuda()
            return reid_loss
        if gt_bbox.dim()==3:
            pred_bboxes_vec = torch.round(pred_bboxes_vec*20).int()
            
            #feat_sz
            reid_loss = torch.tensor(0.0).cuda()

            assert gt_bbox.dim() ==3
            n, b, c, _, _= feature.shape

            vacant1 = torch.zeros(1, 20, 20).cuda()
            vacant2 = torch.zeros(1, 20, 20).cuda()        
            for j in range(self.bs):
                for i in range(n-1):            
                    vacant11 = vacant1.repeat(c, 1, 1)
                    vacant22 = vacant2.repeat(c, 1, 1)

                    if gt_bbox[i, j, 1]>=0 and gt_bbox[i+1, j, 1]>=0:
                        vacant11[:, pred_bboxes_vec[j, i+1, 1]:pred_bboxes_vec[j, i+1, 3],pred_bboxes_vec[j, i+1, 0]:pred_bboxes_vec[j, i+1, 2]] = 1  
                        vacant22[:, pred_bboxes_vec[j, i, 1]:pred_bboxes_vec[j, i, 3],pred_bboxes_vec[j, i, 0]:pred_bboxes_vec[j, i, 2]] = 1  

                        predN_Bbox = feature[i+1, j] * vacant11
                        pred_Bbox = feature[i,j] * vacant22
                        reid_loss += self.objective['reId'](pred_Bbox, predN_Bbox)
                    else:
                        reid_loss += torch.tensor(0.0).cuda()

            return reid_loss/(n*b)
        else:
            reid_loss = torch.tensor(0.0).cuda()
            return reid_loss
