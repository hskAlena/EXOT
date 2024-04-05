from typing import OrderedDict
from pytorch_lightning import LightningModule
from lib.models.exot import build_exotst_odin, build_exotst_cls
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.merge import merge_template_search
import torch
from lib.train.admin import multigpu
#from . import LitEXOTActor
import os


class LitEXOTSTActor(LightningModule):
    def __init__(self, cfg, settings, loss_type, objective, loss_weight, lr =0.0001):
        '''method used to define our model parameters'''
        super().__init__()
        
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.exit_flag = loss_type

        # optimizer parameters
        self.cfg = cfg
        self.lr = lr
        
        self.objective = objective
        self.loss_weight = loss_weight  
        
        if cfg.MODEL.ODIN_TYPE == 'cls':
            net = build_exotst_cls(cfg)
        else:
            net = build_exotst_odin(cfg)
        self.net = net


        # metrics
        # self.accuracy = pl.metrics.Accuracy()

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters(ignore=['cfg', 'settings', 'loss_type', 'objective', 'loss_weight', 'lr'])

    def training_step(self, data, batch_idx, optimizer_idx=None):
        '''needs to return a loss from a single batch'''
        # data, y = batch
        out_dict, flagFeat = self(data, run_box_head=True, run_cls_head=True)
        # process the groundtruth label
        # print(out_dict, data['label'])
        '''
        {'pred_logits': tensor([[[0.1119]],
        [[0.1119]],
        [[0.1134]],
        [[0.0727]],
        [[0.1378]],
        [[0.1219]],
        [[0.1036]],
        [[0.1465]],
        [[0.1339]],
        [[0.1515]],
        [[0.1014]],
        [[0.1466]],
        [[0.0944]],
        [[0.1017]],
        [[0.0865]],
        [[0.1530]]], device='cuda:0', grad_fn=<SelectBackward0>), 'pred_obj': tensor([[[[-0.1900,  0.4178, -0.0296,  ...,  0.0041, -0.4714, -0.0923]],
         [[-0.2813,  0.1830, -0.1882,  ...,  0.2442, -0.2883,  0.2158]],
         [[-0.1125,  0.3063, -0.0026,  ...,  0.2690, -0.4420,  0.0646]],
         ...,
         [[-0.2978,  0.2575, -0.0639,  ...,  0.5419, -0.4117, -0.0342]],
         [[-0.3145,  0.0524, -0.2614,  ...,  0.2964, -0.3564, -0.0841]],
         [[-0.0817,  0.0837, -0.3179,  ...,  0.1379, -0.4021,  0.0880]]]],
       device='cuda:0', grad_fn=<DivBackward0>)} tensor([0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 0.],
       device='cuda:0') tensor([353, 361, 357,   4, 363, 363,   1, 105,   7,   4,  39,  13,  28,   5,
         13,  21], device='cuda:0')

        '''
        

        labels = data['label'].view(-1)  # (batch, ) 0 or 1
        objlabels = data['test_class'].view(-1)
        # print(out_dict, labels, objlabels)
        loss, status = self.compute_losses(out_dict, labels, objlabels)
        gt_bboxes = data['search_anno']
        if gt_bboxes.dim() ==3 and gt_bboxes.shape[0]==1:
            gt_bboxes = gt_bboxes[0]
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

        self.log('Loss/train/total', status['total_loss'])
        self.log('Loss/train/obj', status['obj_loss'])
        self.log('Loss/train/cls', status['cls_loss'])
        self.log('train/obj_accuracy', status['obj_accuracy'])
        self.log('train/IoU', iou['IoU'])
        # self.log('train_batch_stepidx', batch_idx)

        # Log metrics
        return loss

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


    def compute_losses(self, pred_dict, labels, objlabels, return_status=True):
        # print(pred_dict['pred_logits'].shape, pred_dict["pred_logits"].view(-1).shape, labels.shape)
        # print(pred_dict['pred_obj'].shape, torch.squeeze(pred_dict["pred_obj"]).shape, objlabels.shape)
        '''
        torch.Size([16, 1, 1]) torch.Size([16]) torch.Size([16])
        torch.Size([1, 16, 1, 366]) torch.Size([16, 366]) torch.Size([16])
        '''

        clsloss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)
        # print(pred_dict['pred_obj'])
        # print(pred_dict['pred_logits'])
        # print(labels)
        # print(objlabels)
        objloss = self.loss_weight['objcls']*torch.mean(labels*self.objective['objcls'](torch.squeeze(pred_dict["pred_obj"]), objlabels))

        objidx = torch.argmax(torch.squeeze(pred_dict["pred_obj"]).detach(), dim=1)
        objaccuracy = torch.count_nonzero(labels*(objlabels == objidx))/torch.count_nonzero(labels)

        loss = clsloss+objloss
        if return_status:
            # status for log
            status = {
                "cls_loss": clsloss.item(),
                "obj_loss": objloss.item(),
                "total_loss": loss.item(),
                'obj_accuracy': objaccuracy.item()}
            return loss, status
        else:
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


    def validation_step(self, data, batch_idx, optimizer_idx=None):
        '''used for logging metrics'''
        # data, y = batch
        # data['epoch'] = self.epoch
        # data['settings'] = self.settings

        out_dict, flagFeat = self(data, run_box_head=True, run_cls_head=True)
        # process the groundtruth label
        labels = data['label'].view(-1)  # (batch, ) 0 or 1
        objlabels = data['test_class'].view(-1)
        # print(out_dict, labels, objlabels)
        loss, status = self.compute_losses(out_dict, labels, objlabels)
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
    
        self.log('Loss/valid/total', status['total_loss'], sync_dist=True)
        self.log('Loss/valid/obj', status['obj_loss'], sync_dist=True)
        self.log('Loss/valid/cls', status['cls_loss'], sync_dist=True)
        self.log('valid/obj_accuracy', status['obj_accuracy'], sync_dist=True)
        self.log('valid/IoU', iou['IoU'], sync_dist=True)
        # self.log('val_batch_stepidx', batch_idx, sync_dist=True)

        # Log metrics
    
    def test_step(self, data, batch_idx, optimizer_idx=None):
        '''used for logging metrics'''
        # data, y = batch

        out_dict, flagFeat = self(data, run_box_head=True, run_cls_head=True)
        # process the groundtruth label
        labels = data['label'].view(-1)  # (batch, ) 0 or 1
        objlabels = data['test_class'].view(-1)
        # print(out_dict, labels, objlabels)
        loss, status = self.compute_losses(out_dict, labels, objlabels)
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
        self.log('Loss/test/total', status['total_loss'], sync_dist=True)
        self.log('Loss/test/obj', status['obj_loss'], sync_dist=True)
        self.log('Loss/test/cls', status['cls_loss'], sync_dist=True)
        self.log('test/obj_accuracy', status['obj_accuracy'], sync_dist=True)
        self.log('test/IoU', iou['IoU'], sync_dist=True)
        # self.log('test_batch_stepidx', batch_idx, sync_dist=True)

        # Log metrics
        #self.log('test_acc', self.accuracy(logits, y))

    def configure_optimizers(self):
        '''defines model optimizer'''
        if type(self.cfg.TRAIN.SCHEDULER.TYPE) == list:
            self.optimizer, self.lr_scheduler = self.get2_optimizer_scheduler(self.cfg)
        else:
            self.optimizer, self.lr_scheduler = self.get_optimizer_scheduler(self.cfg)  
        # self.optimizer, self.lr_scheduler = self.get_optimizer_scheduler(self.cfg)
        return self.optimizer, self.lr_scheduler

    def get2_optimizer_scheduler(self, cfg):
        train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
        # Adam(self.parameters(), lr=self.lr)
        if train_cls:
            # print("Only training classification head. Learnable parameters are shown below.")
            objcls = []
            cls = []
            for n, p in self.net.named_parameters():
                if n == 'odin_cls.h.weight' or n=='odin_cls.h.bias':
                    objcls.append(p)
                elif 'cls' in n and p.requires_grad:
                    print(n)
                    cls.append(p)
                # elif 'odin_cls' in n and p.requires_grad:
                #     print(n)
                #     cls.append(p)

            objparam_dicts = [
                {"params": objcls}
            ]
            clsparam_dicts = [
                {'params': cls}
            ]

            # print(objparam_dicts)
            # print(clsparam_dicts)

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

        opt = cfg.TRAIN.OPTIMIZER
        optimizer_list = []
        paramlist = [clsparam_dicts, objparam_dicts]
        if type(opt) == list:
            for i in range(len(opt)):
                if opt[i] == "ADAMW":
                    optimizer_list.append(torch.optim.AdamW(paramlist[i], lr=self.lr,
                                                weight_decay=cfg.TRAIN.WEIGHT_DECAY))
                    ## weight decay pick it out.
                elif opt[i] == "SGD":
                    optimizer_list.append(torch.optim.SGD(paramlist[i], lr=0.1, momentum=0.9))
                else:
                    raise ValueError("Unsupported Optimizer")

        lr_scheduler_list = []
        sche = cfg.TRAIN.SCHEDULER.TYPE
        for i in range(len(optimizer_list)):
            if sche[i] == 'step':            
                lr_scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_list[i], cfg.TRAIN.LR_DROP_EPOCH))

            elif sche[i] == "Mstep":
                lr_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(optimizer_list[i],
                                                                    milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                                    gamma=cfg.TRAIN.SCHEDULER.GAMMA))
            else:
                raise ValueError("Unsupported scheduler")
        return optimizer_list, lr_scheduler_list

    def get_optimizer_scheduler(self, cfg):
        train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
        # Adam(self.parameters(), lr=self.lr)
        if train_cls:
            # print("Only training classification head. Learnable parameters are shown below.")
            objcls = []
            cls = []
            for n, p in self.net.named_parameters():
                if "cls" not in n:
                    p.requires_grad = False
                if not cfg.MODEL.CLS_HEAD and 'cls_head' in n:
                    p.requires_grad = False

            for n, p in self.net.named_parameters():
                if 'cls_head' in n and p.requires_grad:
                    cls.append(p)
                elif 'cls' in n and p.requires_grad:
                    objcls.append(p)

            objparam_dicts = [
                {"params": objcls}
            ]
            clsparam_dicts = [
                {'params': cls}
            ]

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

        opt = cfg.TRAIN.OPTIMIZER
        optimizer_list = []
        if cfg.MODEL.CLS_HEAD == False:
            param_dicts = objparam_dicts
        else:
            param_dicts = clsparam_dicts
            param_dicts.append(objparam_dicts[0])
            
        if type(opt) == list:
            paramlist = [clsparam_dicts, objparam_dicts]
            for i in range(len(opt)):
                if opt[i] == "ADAMW":
                    optimizer_list.append(torch.optim.AdamW(paramlist[i], lr=self.lr,
                                                weight_decay=cfg.TRAIN.WEIGHT_DECAY))
                    ## weight decay pick it out.
                elif opt[i] == "SGD":
                    optimizer_list.append(torch.optim.SGD(paramlist[i], lr=self.lr,
                                                weight_decay=cfg.TRAIN.WEIGHT_DECAY, momentum=0.9))
                else:
                    raise ValueError("Unsupported Optimizer")
        else:
            if cfg.TRAIN.OPTIMIZER == "ADAMW":
                optimizer_list = [torch.optim.AdamW(param_dicts, lr=self.lr,
                                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)]
                ## weight decay pick it out.
            elif cfg.TRAIN.OPTIMIZER == "SGD":
                optimizer_list = [torch.optim.SGD(param_dicts, lr=self.lr,
                                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)]
            else:
                raise ValueError("Unsupported Optimizer")

        lr_scheduler_list = []
        
        if cfg.TRAIN.SCHEDULER.TYPE == 'step':
            if type(opt) == list:
                for i in range(len(optimizer_list)):
                    lr_scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_list[i], cfg.TRAIN.LR_DROP_EPOCH))
            else:
                lr_scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_list[0], cfg.TRAIN.LR_DROP_EPOCH))

        elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_list,
                                                                milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                                gamma=cfg.TRAIN.SCHEDULER.GAMMA)
        else:
            raise ValueError("Unsupported scheduler")
        return optimizer_list, lr_scheduler_list


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
        
        '''
        objcls_head.layers.0.weight: copying a param with shape torch.Size([256, 256]) from checkpoint, the shape in current model is torch.Size([5120, 102400]).
        size mismatch for objcls_head.layers.0.bias: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([5120]).
        size mismatch for objcls_head.layers.1.weight: copying a param with shape torch.Size([256, 256]) from checkpoint, the shape in current model is torch.Size([5120, 5120]).
        size mismatch for objcls_head.layers.1.bias: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([5120]).
        size mismatch for objcls_head.h.weight: copying a param with shape torch.Size([366, 256]) from checkpoint, the shape in current model is torch.Size([366, 5120]).
        size mismatch for objcls_head.g.weight:
        '''
        try:
            missing_k, unexpected_k = net.load_state_dict(checkpoint["net"], strict=False)
        except:
            ckptitem = checkpoint['net'].items()
            net_kv = OrderedDict()
            count = 0
            for key, value in ckptitem:
                if 'objcls' in key:
                    continue
                # name = key.replace('objcls_head', 'objcls')
                net_kv[key] = value
            missing_k, unexpected_k = net.load_state_dict(net_kv, strict=False)
        # net.eval() ?? 
        # missing_k, unexpected_k = net.load_state_dict(net_kv, strict=False)

        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)
        # missing_k, unexpected_k = self.optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
        # print("previous checkpoint is loaded.")
        # print("missing keys: ", missing_k)
        # print("unexpected keys:", unexpected_k)

    # def training_epoch_end(self, training_step_outputs):
    #     """Saves a checkpoint of the network and other variables."""

    #     net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net

    #     if self.settings.env.workspace_dir is not None:
    #         self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
    #         '''2021.1.4 New function: specify checkpoint dir'''
    #         if self.settings.save_dir is None:
    #             self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
    #         else:
    #             self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
    #         print("checkpoints will be saved to %s" % self._checkpoint_dir)

    #         if self.settings.local_rank in [-1, 0]:
    #             if not os.path.exists(self._checkpoint_dir):
    #                 print("Training with multiple GPUs. checkpoints directory doesn't exist. "
    #                       "Create checkpoints directory")
    #                 os.makedirs(self._checkpoint_dir)
    #     else:
    #         self._checkpoint_dir = None

    #     # actor_type = type(actor).__name__
    #     net_type = type(net).__name__
    #     state = {
    #         'epoch': self.current_epoch,
    #         # 'actor_type': actor_type,
    #         'net_type': net_type,
    #         'net': net.state_dict(),
    #         'net_info': getattr(net, 'info', None),
    #         'constructor': getattr(net, 'constructor', None),
    #         # 'optimizer': self.optimizer.state_dict(),
    #         'stats': training_step_outputs,
    #         'settings': self.settings
    #     }

    #     directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
    #     print(directory)
    #     if not os.path.exists(directory):
    #         print("directory doesn't exist. creating...")
    #         os.makedirs(directory)

    #     # First save as a tmp file
    #     tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.current_epoch)
    #     torch.save(state, tmp_file_path)

    #     file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.current_epoch)

    #     # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
    #     os.rename(tmp_file_path, file_path)
