from re import L
from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search


class EXOTMergeActor(BaseActor):
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
        out_dict, flagFeat = self.forward_pass(data, run_box_head=True, run_cls_head=True)

        # process the groundtruth
        labels = data['label'].view(-1)  # (batch, ) 0 or 1
        objlabels = data['test_class'].view(-1)

        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        if gt_bboxes.dim() ==3 and gt_bboxes.shape[0]==1:
            gt_bboxes = gt_bboxes[0]

        loss, status = self.compute_losses(out_dict, gt_bboxes, labels, objlabels, flag_feat = flagFeat )
        
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

    def compute_losses(self, pred_dict, gt_bbox, labels, objlabels, flag_feat = None, return_status=True):
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

        clsloss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)
        objloss = self.loss_weight['objcls']*torch.mean(labels*self.objective['objcls'](torch.squeeze(pred_dict["pred_obj"]), objlabels))
        objidx = torch.argmax(torch.squeeze(pred_dict["pred_obj"]).detach(), dim=1)
        objaccuracy = torch.count_nonzero(labels*(objlabels == objidx))/torch.count_nonzero(labels)

        loss += clsloss+objloss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()

            status = {"Loss/total": loss.item(),
                          "Loss/giou": giou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "IoU": mean_iou.item(),
                        "Loss/cls_loss": clsloss.item(),
                "Loss/obj_loss": objloss.item(),
                "total_loss": loss.item(),
                'obj_accuracy': objaccuracy.item()}
            return loss, status
        else:
            return loss
