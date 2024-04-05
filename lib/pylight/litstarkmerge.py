from os import sync
from pytorch_lightning import LightningModule
from lib.models.stark import build_starkst
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.merge import merge_template_search
import torch
from lib.train.admin import multigpu

class LitSTARKMergeActor(LightningModule):
    def __init__(self, cfg, settings, loss_type, objective, loss_weight, lr =0.0001):
        '''method used to define our model parameters'''
        super().__init__()    
        
        self.lr = lr
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.exit_flag = loss_type

        # optimizer parameters
        self.cfg = cfg
        self.objective = objective
        self.loss_weight = loss_weight

        net = build_starkst(cfg)
        self.net = net
        
        # metrics
        # self.accuracy = pl.metrics.Accuracy()

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters(ignore=['cfg', 'settings', 'loss_type', 'objective', 'loss_weight', 'lr'])

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
        out_dict, _, _ = self.net(seq_dict=seq_dict, mode="transformer", run_box_head=run_box_head, run_cls_head=run_cls_head)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def training_step(self, data, batch_idx):
        '''needs to return a loss from a single batch'''
        # data, y = batch
        # forward pass
        out_dict = self(data, run_box_head=True, run_cls_head=True)

        # process the groundtruth
        gt_bboxes = data['search_anno']
        if gt_bboxes.dim() ==3 and gt_bboxes.shape[0]==1:
            gt_bboxes = gt_bboxes[0]
        labels = data['label'].view(-1)  # (batch, ) 0 or 1
        objlabels = data['test_class'].view(-1)
        # print(out_dict, labels, objlabels)
        loss, status = self.compute_losses(out_dict, gt_bboxes, labels)
        
        iou = self.compute_iou(out_dict, gt_bboxes)

        conf_score = out_dict["pred_logits"].view(-1).sigmoid()
        conf_tf = conf_score>0.5
        conf_neg = torch.count_nonzero(conf_score<0.5)
        conf_num = torch.count_nonzero(conf_tf == labels)
        conf_gtneg = torch.count_nonzero(1-labels)
        
        # return loss, status
        log_out = {
            'log_mean': torch.mean(out_dict['pred_logits']).item(),
            'log_min': torch.min(out_dict['pred_logits']).item(),
            'log_max': torch.max(out_dict['pred_logits']).item()}

        # Log training loss
        self.log('train/logit_mean', log_out['log_mean'])
        self.log('train/logit_min', log_out['log_min'])
        self.log('train/logit_max', log_out['log_max'])

        self.log('train/logit_accuracy', conf_num)
        self.log('train/logit_predneg', conf_neg)
        self.log('train/logit_gtneg', conf_gtneg)

        self.log('Loss/train/cls', status['cls_loss'])
        self.log('train/IoU', iou['IoU'])

        return loss

    def compute_iou(self, pred_dict, gt_bbox, return_status=True):
        pred_boxes = pred_dict['pred_boxes'].detach()
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_bboxes_vec = box_cxcywh_to_xyxy(pred_boxes)
        pred_boxes_vec = pred_bboxes_vec.view(-1, 4) # (B,N,4) --> (BN,4) (x1,y1,x2,y2)

        if gt_bbox.dim() ==3:
            gt_bboxes_vec = box_xywh_to_xyxy(gt_bbox.detach()).clamp(min=-1.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            gt_boxes_vec = gt_bboxes_vec.view(-1,4).clamp(min=0.0, max=1.0)            
        else:
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox.detach())[:, None, :]
            gt_bboxes_vec = gt_boxes_vec.repeat((1, num_queries, 1)).clamp(min=-1.0, max=1.0)
            gt_boxes_vec = gt_bboxes_vec.view(-1, 4).clamp(min=0.0, max=1.0)              # (B,4) --> (B,1,4) --> (B,N,4)
            
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).type_as(pred_boxes), torch.tensor(0.0).type_as(pred_boxes)

        # status for log
        mean_iou = iou.detach().mean()
        status = { "IoU": mean_iou.item()}
        return status

    def validation_step(self, data, batch_idx):
        '''used for logging metrics'''
        # data, y = batch
        # data['epoch'] = self.epoch
        # data['settings'] = self.settings

        out_dict = self(data, run_box_head=False, run_cls_head=True)

        # process the groundtruth
        labels = data['label'].view(-1)  # (batch, ) 0 or 1

        objlabels = data['test_class'].view(-1)
        # print(out_dict, labels, objlabels)
        loss, status = self.compute_losses(out_dict, labels)
        gt_bboxes = data['search_anno']
        if gt_bboxes.dim() ==3 and gt_bboxes.shape[0]==1:
            gt_bboxes = gt_bboxes[0]
        iou = self.compute_iou(out_dict, gt_bboxes)

        conf_score = out_dict["pred_logits"].view(-1).sigmoid()
        conf_tf = conf_score>0.5
        conf_neg = torch.count_nonzero(conf_score<0.5)
        conf_num = torch.count_nonzero(conf_tf == labels)
        conf_gtneg = torch.count_nonzero(1-labels)

        # Log validation loss (will be automatically averaged over an epoch)
        # Log training loss
        log_out = {
            'log_mean': torch.mean(out_dict['pred_logits']).item(),
            'log_min': torch.min(out_dict['pred_logits']).item(),
            'log_max': torch.max(out_dict['pred_logits']).item()
        }

        # Log training loss
        self.log('valid/logit_mean', log_out['log_mean'], sync_dist=True)
        self.log('valid/logit_min', log_out['log_min'], sync_dist=True)
        self.log('valid/logit_max', log_out['log_max'], sync_dist=True)

        self.log('valid/logit_accuracy', conf_num, sync_dist = True)
        self.log('valid/logit_predneg', conf_neg, sync_dist = True)
        self.log('valid/logit_gtneg', conf_gtneg, sync_dist = True)
    
        self.log('Loss/valid/cls', status['cls_loss'], sync_dist=True)
        self.log('valid/IoU', iou['IoU'], sync_dist=True)
    
    def test_step(self, data, batch_idx):
        '''used for logging metrics'''
        # data, y = batch

        out_dict = self(data, run_box_head=False, run_cls_head=True)

        # process the groundtruth
        labels = data['label'].view(-1)  # (batch, ) 0 or 1

        objlabels = data['test_class'].view(-1)
        # print(out_dict, labels, objlabels)
        loss, status = self.compute_losses(out_dict, labels)
        gt_bboxes = data['search_anno']
        if gt_bboxes.dim() ==3 and gt_bboxes.shape[0]==1:
            gt_bboxes = gt_bboxes[0]
        iou = self.compute_iou(out_dict, gt_bboxes)

        conf_score = out_dict["pred_logits"].view(-1).sigmoid()
        conf_tf = conf_score>0.5
        conf_neg = torch.count_nonzero(conf_score<0.5)
        conf_num = torch.count_nonzero(conf_tf == labels)
        conf_gtneg = torch.count_nonzero(1-labels)

        log_out = {
             'log_mean': torch.mean(out_dict['pred_logits']).item(),
            'log_min': torch.min(out_dict['pred_logits']).item(),
            'log_max': torch.max(out_dict['pred_logits']).item()
        }

        # Log training loss
        self.log('test/logit_mean', log_out['log_mean'], sync_dist=True)
        self.log('test/logit_min', log_out['log_min'], sync_dist=True)
        self.log('test/logit_max', log_out['log_max'], sync_dist=True)

        self.log('test/logit_accuracy', conf_num, sync_dist = True)
        self.log('test/logit_predneg', conf_neg, sync_dist = True)
        self.log('test/logit_gtneg', conf_gtneg, sync_dist = True)

        # Log test loss
        self.log('Loss/test/cls', status['cls_loss'], sync_dist=True)
        self.log('test/IoU', iou['IoU'], sync_dist=True)

        # Log metrics
        #self.log('test_acc', self.accuracy(logits, y))
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        self.optimizer, self.lr_scheduler = self.get_optimizer_scheduler(self.cfg)
        return [self.optimizer], [self.lr_scheduler]

    def compute_losses(self, pred_dict, labels, return_status=True):
        loss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)
        if return_status:
            # status for log
            status = {
                "cls_loss": loss.item()}
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
            param_dicts = [
                {"params": [p for n, p in self.net.named_parameters() if "backbone" not in n and p.requires_grad]},
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
        checkpoint['optimizer'] = self.optimizer.state_dict()

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
