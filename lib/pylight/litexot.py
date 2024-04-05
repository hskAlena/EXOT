from pytorch_lightning import LightningModule
from lib.models.exot import build_exotst_odin, build_exotst_cls
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.merge import merge_template_search
import torch
from lib.train.admin import multigpu

class LitEXOTActor(LightningModule):
    def __init__(self, cfg, settings, loss_type, objective, loss_weight, lr =0.0001):
        '''method used to define our model parameters'''
        super().__init__()
        
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.exit_flag = loss_type

        # optimizer parameters
        self.cfg = cfg
        self.lr = lr

        if cfg.MODEL.ODIN_TYPE == 'cls':
            net = build_exotst_cls(cfg)
        else:
            net = build_exotst_odin(cfg)
        self.net = net
        self.objective = objective
        self.loss_weight = loss_weight

        # metrics
        # self.accuracy = pl.metrics.Accuracy()

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B

        self.save_hyperparameters(ignore=['cfg', 'settings', 'objective', 'loss_weight', 'lr'])
       

    def forward(self, data, run_box_head=True, run_cls_head=False):
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
        seq_dict = merge_template_search(feat_dict_list)
        
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

    def training_step(self, data, batch_idx):
        '''needs to return a loss from a single batch'''
        # data, y = batch

        out_dict, flagFeat = self(data, run_box_head=True, run_cls_head=False)
        gt_exit, gt_package, gt_bboxes = self.process_gt(data)

        # compute losses
        # if flagFeat == None:
        #     flagFeat = (gt_exit, gt_package) #, data['epoch'])
        # else:
        #     exitflag, feature = flagFeat
        #     flagFeat = (gt_exit, gt_package, feature, exitflag) #, data['epoch'])
        loss, status = self.compute_losses(out_dict, gt_bboxes, flag_feat = flagFeat)
        
        # return loss, status

        # Log training loss
        self.log('Loss/train_total', loss)
        # self.log('train_batch_stepidx', batch_idx)

        # Log metrics

        self.log('Loss/train_giou', status['Loss/giou'])
        self.log('Loss/train_l1', status['Loss/l1'])
        self.log('train_IoU', status['IoU'])


        return loss

    def validation_step(self, data, batch_idx):
        '''used for logging metrics'''
        # data, y = batch
        # data['epoch'] = self.epoch
        # data['settings'] = self.settings

        out_dict, flagFeat = self(data, run_box_head=True, run_cls_head=False)
        gt_exit, gt_package, gt_bboxes = self.process_gt(data)

        # compute losses
        # if flagFeat == None:
        #     flagFeat = (gt_exit, gt_package) #, data['epoch'])
        # else:
        #     exitflag, feature = flagFeat
        #     flagFeat = (gt_exit, gt_package, feature, exitflag) #, data['epoch'])
        loss, status = self.compute_losses(out_dict, gt_bboxes, flag_feat = flagFeat)

        # Log validation loss (will be automatically averaged over an epoch)
        # Log training loss
        self.log('Loss/valid_total', loss, sync_dist=True)
        # self.log('val_batch_stepidx', batch_idx, sync_dist=True)

        # Log metrics

        self.log('Loss/valid_giou', status['Loss/giou'], sync_dist=True)
        self.log('Loss/valid_l1', status['Loss/l1'], sync_dist=True)
        self.log('valid_IoU', status['IoU'], sync_dist=True)
    
    def test_step(self, data, batch_idx):
        '''used for logging metrics'''
        # data, y = batch

        out_dict, flagFeat = self(data, run_box_head=True, run_cls_head=False)
        gt_exit, gt_package, gt_bboxes = self.process_gt(data)

        # compute losses
        # if flagFeat == None:
        #     flagFeat = (gt_exit, gt_package) #, data['epoch'])
        # else:
        #     exitflag, feature = flagFeat
        #     flagFeat = (gt_exit, gt_package, feature, exitflag) #, data['epoch'])
        loss, status = self.compute_losses(out_dict, gt_bboxes, flag_feat = flagFeat)

        # Log test loss
        self.log('Loss/test_total', loss, sync_dist=True)
        # self.log('test_batch_stepidx', batch_idx, sync_dist=True)

        # Log metrics
        self.log('Loss/test_giou', status['Loss/giou'], sync_dist=True)
        self.log('Loss/test_l1', status['Loss/l1'], sync_dist=True)
        self.log('test_IoU', status['IoU'], sync_dist=True)

        # Log metrics
        #self.log('test_acc', self.accuracy(logits, y))
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        self.optimizer, self.lr_scheduler = self.get_optimizer_scheduler(self.cfg)
        return [self.optimizer], [self.lr_scheduler]

    def process_gt(self, data):
        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # if data['joint_flag'][0] != 'None':            
        #     gt_joints = torch.squeeze(data['search_joint'])
        #     gt_depth = torch.squeeze(data['search_depth'])
        #     template_depth = data['template_depth']
        #     template_bboxes = box_xywh_to_xyxy(data['template_anno']) 
        #     gt_package = [gt_joints, gt_depth, template_depth, template_bboxes]
        # else:
        template_bboxes = box_xywh_to_xyxy(data['template_anno']) 
        gt_package = [template_bboxes]


        gt_exit = torch.squeeze(data['search_exit'])
        # compute losses
        if gt_bboxes.dim() ==3 and gt_bboxes.shape[0]==1:
            gt_bboxes = gt_bboxes[0]
        return gt_exit, gt_package, gt_bboxes

    def compute_losses(self, pred_dict, gt_bbox,flag_feat = None, return_status=True):
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
        
        # clsloss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)
        # objloss = self.loss_weight['objcls']*torch.mean(labels*self.objective['objcls'](torch.squeeze(pred_dict["pred_obj"]), objlabels))

        # objidx = torch.argmax(torch.squeeze(pred_dict["pred_obj"]).detach(), dim=1)
        # objaccuracy = torch.count_nonzero(labels*(objlabels == objidx))/torch.count_nonzero(labels)

        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).type_as(pred_boxes), torch.tensor(0.0).type_as(pred_boxes)
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

    def get_optimizer_scheduler(self, cfg):
        train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
        # Adam(self.parameters(), lr=self.lr)
        if train_cls:
            # print("Only training classification head. Learnable parameters are shown below.")
            param_dicts = [
                {"params": [p for n, p in self.net.named_parameters() if "cls" in n and p.requires_grad]}
            ]

            for n, p in self.net.named_parameters():
                if "cls" not in n:
                    p.requires_grad = False
                # else:
                #     print(n)
        else:
            tmplist = []
            if cfg.MODEL.CLS_HEAD:
                tmplist = [p for n, p in self.net.named_parameters() if "backbone" not in n and p.requires_grad]
            else:
                for n, p in self.net.named_parameters():
                    if "backbone" not in n and 'cls_head' not in n and p.requires_grad:
                        tmplist.append(p)

            param_dicts = [
                #{"params": [p for n, p in self.net.named_parameters() if "backbone" not in n and p.requires_grad]},
                {"params": tmplist},
                {
                    "params": [p for n, p in self.net.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.lr * cfg.TRAIN.BACKBONE_MULTIPLIER,
                },
            ]
            # if is_main_process():
            #     print("Learnable parameters are shown below.")
            #     for n, p in self.net.named_parameters():
            #         if p.requires_grad:
            #             print(n)

        if cfg.TRAIN.OPTIMIZER == "ADAMW":
            optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                        weight_decay=cfg.TRAIN.WEIGHT_DECAY)
            ## weight decay pick it out.
        elif cfg.TRAIN.OPTIMIZER == "SGD":
            optimizer = torch.optim.SGD(param_dicts, lr=self.lr,
                                        weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        else:
            raise ValueError("Unsupported Optimizer")
        if cfg.TRAIN.SCHEDULER.TYPE == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
        elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                                gamma=cfg.TRAIN.SCHEDULER.GAMMA)
        else:
            raise ValueError("Unsupported scheduler")
        return optimizer, lr_scheduler


    def on_save_checkpoint(self, checkpoint):
        checkpoint["net"] = self.net.state_dict()
        ignore_fields = ['settings','lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info']
        # checkpoint['optimizer'] = self.optimizer.state_dict()

        '''        
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }'''

    def on_load_checkpoint(self, checkpoint):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        missing_k, unexpected_k = net.load_state_dict(checkpoint["net"], strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)
        # missing_k, unexpected_k = self.optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
        # print("previous checkpoint is loaded.")
        # print("missing keys: ", missing_k)
        # print("unexpected keys:", unexpected_k)
