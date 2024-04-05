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

def main():
    args = parse_args()
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

    nums_folds = 5
    for k in range(nums_folds):
        wandb_logger = WandbLogger(project='EXOT', name=f"{settings.script_name}_{settings.config_name}_fold_{k}/{nums_folds}")
        if settings.script_name == "exot_st1" or settings.script_name == "exot_merge":
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/train/{settings.script_name}/{settings.config_name}_fold_{k}_{nums_folds}/", filename='EXOTST_{epoch}', every_n_epochs=1)
        elif settings.script_name == "exot_st2":
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/train/{settings.script_name}/{settings.config_name}_fold_{k}_{nums_folds}/{args.st1_name.split('.')[0]}/", filename='EXOTST_{epoch}', every_n_epochs=1)
        elif settings.script_name == "stark_st1":
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/train/{settings.script_name}/{settings.config_name}_fold_{k}_{nums_folds}/", filename='STARKST_{epoch}', every_n_epochs=1)
        elif settings.script_name == "stark_st2":
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/train/{settings.script_name}/{settings.config_name}_fold_{k}_{nums_folds}/", filename='STARKST_{epoch}', every_n_epochs=1)
        checkpoint_callback.FILE_EXTENSION='.pth.tar'
        
        if args.mode=='single':
            device = 1
            trainer = Trainer(logger=wandb_logger, accelerator='gpu', devices=device, max_epochs=cfg.TRAIN.EPOCH, callbacks=[checkpoint_callback])

        else:
            device = 2
            trainer = Trainer(logger=wandb_logger, accelerator='gpu', strategy='dp', devices=device, max_epochs=cfg.TRAIN.EPOCH, callbacks=[checkpoint_callback])
        #trainer = Trainer(logger=wandb_logger, accelerator='gpu', strategy='dp', devices=device, max_epochs=cfg.TRAIN.EPOCH, callbacks=[checkpoint_callback])
        robot_data = RobotDataModule(data_dir='data/robot-data/', k=k, num_splits=nums_folds, batch_size=cfg.TRAIN.BATCH_SIZE, kfoldness = True, test_size=0.4)
        robot_data.fill_state(cfg, settings)
        
        # print(model.net)
        if settings.script_name == 'exot_st2':
            model = LitEXOTSTActor(cfg, settings, loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR) 
            # checkpoint = torch.load(f"checkpoints/train/exot_st1/baseline_mix_fold{k}/EXOTST_epepoch=0.pth.tar", map_location='cpu')
            model = LitEXOTSTActor.load_from_checkpoint(f"checkpoints/train/{args.script_prv}/{args.st1_name}", cfg=cfg, settings=settings, \
            loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR, strict=False)
            # if args.resume:
            #     model = LitEXOTSTActor.load_from_checkpoint(f"checkpoints/train/{settings.script_name}/{args.resume_name}", cfg=cfg, settings=settings, \
            #     loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR, strict=False)
            # else:
            #     model = LitEXOTSTActor.load_from_checkpoint(f"checkpoints/train/{args.script_prv}/{args.st1_name}", cfg=cfg, settings=settings, \
            #     loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR, strict=False)
            # model.re_load(cfg, settings, loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR)
            '''
            "epoch", "global_step", "pytorch-lightning_version", "state_dict", 
            "loops", "callbacks", "optimizer_states", "lr_schedulers", 
            "hparams_name", "hyper_parameters".
            '''
        elif settings.script_name == 'exot_st1':
            model = LitEXOTActor(cfg, settings, loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR)
            # if wandb.run.resumed:
            #     model = LitEXOTActor.load_from_checkpoint(f"checkpoints/train/{settings.script_name}/{settings.config_name}_fold_0_5/EXOTST_epoch=23.pth.tar", strict=True)
            if args.resume:
                model = LitEXOTActor.load_from_checkpoint(f"checkpoints/train/{settings.script_name}/{args.resume_name}", cfg=cfg, settings=settings, \
                loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR, strict=False)
        elif settings.script_name == 'exot_merge':
            model = LitEXOTMergeActor(cfg, settings, loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR)
            if args.resume:
                model = LitEXOTMergeActor.load_from_checkpoint(f"checkpoints/train/{settings.script_name}/{args.resume_name}", cfg=cfg, settings=settings, \
                loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR, strict=False)

        elif settings.script_name == 'stark_st2':
            model = LitSTARKSTActor(cfg, settings, loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR)
            if args.resume:
                model = LitSTARKSTActor.load_from_checkpoint(f"checkpoints/train/{settings.script_name}/{args.resume_name}", cfg=cfg, settings=settings, \
                loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR, strict=False)
            else:
                model = LitSTARKSTActor.load_from_checkpoint(f"checkpoints/train/{args.script_prv}/{args.st1_name}", cfg, settings, \
                loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR, strict=False)
        elif settings.script_name == 'stark_st1':
            model = LitSTARKActor(cfg, settings, loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR)
            # model.fill_hyperparam(cfg, settings, loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight)
            if args.resume:
                model = LitSTARKActor.load_from_checkpoint(f"checkpoints/train/{settings.script_name}/{args.resume_name}", cfg=cfg, settings=settings, \
                loss_type=cfg.MODEL.LOSS_TYPE, objective=objective, loss_weight = loss_weight, lr=cfg.TRAIN.LR, strict=False)

        if args.resume:            
            if settings.script_name == "exot_st2":
                trainer.fit(model, datamodule=robot_data, ckpt_path = f"checkpoints/train/{settings.script_name}/{args.config}_fold_0_5/{args.config_prv}_fold_0_5/{args.resume_name}")
            else:
                trainer.fit(model, datamodule=robot_data, ckpt_path = f"checkpoints/train/{settings.script_name}/{args.resume_name}")
        else:
            trainer.fit(model, datamodule=robot_data)
        wandb.save(os.path.join(wandb.run.dir, "%s_%s_model_kfold_%d_%d.pth.tar"%(settings.script_name, settings.config_name, k, nums_folds)))
        #self.log(..., batch_size=batch_size)
        wandb.finish()
        ## ODIN batchwise - classification diff objects possible?
        break

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    parser.add_argument('--seed', type=int, default=42, help='seed for random numbers')
    parser.add_argument('--dry_run', type=int, default=1, help='0: wandb activate, 1: wandb off')
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--st1_name', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--resume_name', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--resume', type=bool, default=False, help='resume from previous checkpoint resume_name.')


    args = parser.parse_args()

    return args

def parse_args_jup(args):
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    parser.add_argument('--seed', type=int, default=42, help='seed for random numbers')
    parser.add_argument('--dry_run', type=int, default=1, help='0: wandb activate, 1: wandb off')
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--st1_name', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--resume_name', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--resume', type=bool, default=False, help='resume from previous checkpoint resume_name.')

    args = parser.parse_args(args)

    return args

class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()       

    def set_default(self):
        self.env = env_settings()
        self.use_gpu = True

    def set_args(self, args):
        # self.args = args
        self.script_name = args.script
        self.config_name = args.config
        self.dry_run = args.dry_run
        self.project_path = 'train/{}/{}'.format(self.script_name, self.config_name)
        if args.script_prv is not None and args.config_prv is not None:
            self.project_path_prv = 'train/{}/{}'.format(args.script_prv, args.config_prv)
        self.local_rank = args.local_rank
        self.save_dir = os.path.abspath(args.save_dir)
        self.use_lmdb = args.use_lmdb
        self.mode = args.mode
        prj_dir = os.path.abspath('')
        # prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (self.script_name, self.config_name))

        self.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

        # update the default configs with config file
        if not os.path.exists(self.cfg_file):
            raise ValueError("%s doesn't exist." % self.cfg_file)
        config_module = importlib.import_module("lib.config.%s.config" % self.script_name)
        cfg = config_module.cfg
        config_module.update_config_from_file(self.cfg_file)     

        # Record the training log
        log_dir = os.path.join(self.save_dir, 'logs')
        if self.local_rank in [-1, 0]:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, "%s-%s.log" % (self.script_name, self.config_name))
        return cfg

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
