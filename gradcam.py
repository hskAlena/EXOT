from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger

# Pytorch modules
import torch

# Pytorch-Lightning
from pytorch_lightning import Trainer
import argparse
from lib.train.admin.environment import env_settings
import os
import numpy as np
import random
import importlib
import cv2 as cv
from lib.train.base_functions import *

from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
# train pipeline related
from lib.train.base_functions import *

# forward propagation related
import importlib
# from sklearn.model_selection import KFold
from lib.train.admin import multigpu
from pytorch_lightning.callbacks import ModelCheckpoint
from lib.pylight import LitEXOTActor, LitEXOTMergeActor, LitEXOTSTActor, RobotDataModule, LitSTARKActor, LitSTARKSTActor
from pl_prac import Settings, init_seeds, parse_args_jup

from typing import OrderedDict
from pytorch_lightning import LightningModule
from lib.models.exot import build_exotst_odin, build_exotst_cls


args = ["--script", "exot_st2", '--config', 'cos_mix_lowdim', '--save_dir', '.', '--mode', 'single', 
'--script_prv', 'exot_st2', '--config_prv', 'cos_mix_lowdim', '--st1_name', 
"cos_mix_lowdim_fold_0_5/cos_mix_lowdim_fold_0_5/EXOTST_epoch=250-v1/EXOTST_epoch=71.pth.tar"]
args = parse_args_jup(args)
cv.setNumThreads(0)
torch.backends.cudnn.benchmark = args.cudnn_benchmark

print('script_name: {}.py  config_name: {}.yaml'.format(args.script, args.config))

'''2021.1.5 set seed for different process'''
if args.seed is not None:
    if args.local_rank != -1:
        init_seeds(args.seed + args.local_rank)
    else:
        init_seeds(args.seed)

settings = Settings()
cfg = settings.set_args(args)
cfg.TRAIN.BATCH_SIZE = 1
# update settings based on cfg
if settings.local_rank in [-1, 0]:
    print("New configuration is shown below.")
    for key in cfg.keys():
        print("%s configuration:" % key, cfg[key])
        print('\n')  
update_settings(settings, cfg)

if settings.script_name == "exot_st2" or  settings.script_name == "exot_merge":
    objective = {'giou': giou_loss, 'l1': l1_loss, 'cls': BCEWithLogitsLoss(), 'objcls': CrossEntropyLoss()}  #reduction='none'
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'cls': cfg.TRAIN.CLS_WEIGHT, 'objcls': cfg.TRAIN.OBJCLS_WEIGHT}
elif settings.script_name == "exot_st1":
    objective = {'giou': giou_loss, 'l1': l1_loss}  #reduction='none'
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
elif settings.script_name == "stark_s" or settings.script_name == "stark_st1":
    objective = {'giou': giou_loss, 'l1': l1_loss}
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}        
elif settings.script_name == "stark_st2":
    objective = {'giou': giou_loss, 'cls': BCEWithLogitsLoss()}
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'cls': 1.0}


if cfg.MODEL.ODIN_TYPE == 'cls':
    net = build_exotst_cls(cfg)
else:
    net = build_exotst_odin(cfg)

net = net.module if multigpu.is_multi_gpu(net) else net
        
checkpoint = torch.load(f"checkpoints/train/{args.script_prv}/{args.st1_name}")

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

net.eval()


k=0
nums_folds = 5
robot_data = RobotDataModule(data_dir='data/robot-data/', k=k, num_splits=nums_folds, batch_size=cfg.TRAIN.BATCH_SIZE, kfoldness = True, test_size=0.4)
robot_data.fill_state(cfg, settings)
robot_data.prepare_data()
robot_data.setup(stage='fit')


from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.merge import merge_template_search
count = 0
# cam = GradCAM(model=model, target_layers=target_layers,
#                 #use_cuda=args.use_cuda
#                 )
for data in robot_data.val_dataloader():
    data = data
    break


feat_dict_list = []
# process the templates
for i in range(2):
    template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
    template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
    feat_dict_list.append(net.forward_backbone(NestedTensor(template_img_i, template_att_i)))

# process the search regions (t-th frame)
search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
feat_dict_list.append(net.forward_backbone(NestedTensor(search_img, search_att)))

# run the transformer and compute losses
seq_dict = merge_template_search(feat_dict_list, return_search=True, return_template=True)

template_bboxes = box_xywh_to_xyxy(data['template_anno'])  #(N_t, batch, 4)
template_joint = None
annot = (template_bboxes, template_joint)


output_embed, enc_mem = net.transformer(seq_dict["feat"], seq_dict["mask"], net.query_embed.weight,
                                            seq_dict["pos"], return_encoder_output=True)

batch_feat = seq_dict['feat_x'].permute((1, 0, 2))

if batch_feat.dim() <2:
    batch_feat = torch.unsqueeze(batch_feat, 0)

batch_feat = batch_feat.permute(0,2,1)

x = (batch_feat, output_embed, enc_mem, annot)


target_layers = [net.odin_cls.reduceNet[-1]]
input_tensor = x
# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)

# That are, for example, combinations of categories, or specific outputs in a non standard model.

targets = [ClassifierOutputTarget(data['test_class'].view(-1))]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]

orig_image = torch.squeeze(data['search_images']).permute(1,2,0)
orig_image -= torch.min(orig_image)
orig_image /= torch.max(orig_image)


visualization = show_cam_on_image(orig_image, grayscale_cam, use_rgb=True)
