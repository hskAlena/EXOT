import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.stark import build_starks, build_starkst
from lib.models.exot import build_exotst_odin, build_exotst_cls
from lib.models.stark import build_stark_lightning_x_trt
# forward propagation related
from lib.train.actors import STARKSActor, STARKSTActor
from lib.train.actors import EXOTActor, EXOTSTActor, EXOTMergeActor
from lib.train.actors import STARKLightningXtrtActor
# for import modules
import importlib


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "stark_s":
        net = build_starks(cfg)
    elif settings.script_name == "stark_st1" or settings.script_name == "stark_st2":
        net = build_starkst(cfg)
    elif settings.script_name == "stark_lightning_X_trt":
        net = build_stark_lightning_x_trt(cfg, phase="train")
    elif settings.script_name in ["exot_st1", "exot_st2", 'exot_merge']:
        if cfg.MODEL.ODIN_TYPE == 'cls':
            net = build_exotst_cls(cfg)
        else:
            net = build_exotst_odin(cfg)    
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    
    if settings.dry_run==0:
        import wandb
        wandb.tensorboard.patch(root_logdir="logs/")
        wandb.init(project="STARK_train",
                sync_tensorboard=True, reinit=True, config=cfg.TRAIN)
        print("WANDB CONFIG", cfg.TRAIN, "\n", wandb.config)
        w_cfg = wandb.config

        # update settings based on cfg
        update_set_afterwandb(settings, w_cfg.GRAD_CLIP_NORM)
        wandb.watch(net, log='all')
    print("DRY run", settings.dry_run)
    
    # Loss functions and Actors    
    if settings.script_name == "stark_s" or settings.script_name == "stark_st1":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKSActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == "stark_st2":
        objective = {'cls': BCEWithLogitsLoss()}
        loss_weight = {'cls': 1.0}
        actor = STARKSTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == "exot_st1":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        if settings.dry_run==1:
            loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        else:
            loss_weight = {'giou': w_cfg.GIOU_WEIGHT, 'l1': w_cfg.L1_WEIGHT}
        actor = EXOTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, loss_type=cfg.MODEL.LOSS_TYPE)
    elif settings.script_name == "exot_st2":
        objective = {'giou': giou_loss, 'l1': l1_loss, 'cls': BCEWithLogitsLoss(), 'objcls': CrossEntropyLoss()} 
        if settings.dry_run==1:
            loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'cls': cfg.TRAIN.CLS_WEIGHT, 'objcls': cfg.TRAIN.OBJCLS_WEIGHT}
        else:
            loss_weight = {'giou': w_cfg.GIOU_WEIGHT, 'l1': w_cfg.L1_WEIGHT, 'cls': cfg.TRAIN.CLS_WEIGHT, 'objcls': cfg.TRAIN.OBJCLS_WEIGHT}
        actor = EXOTSTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, loss_type=cfg.MODEL.LOSS_TYPE)
    elif settings.script_name == "exot_merge":
        objective = {'giou': giou_loss, 'l1': l1_loss, 'cls': BCEWithLogitsLoss(), 'objcls': CrossEntropyLoss()} 
        if settings.dry_run==1:
            loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'cls': cfg.TRAIN.CLS_WEIGHT, 'objcls': cfg.TRAIN.OBJCLS_WEIGHT}
        else:
            loss_weight = {'giou': w_cfg.GIOU_WEIGHT, 'l1': w_cfg.L1_WEIGHT, 'cls': cfg.TRAIN.CLS_WEIGHT, 'objcls': cfg.TRAIN.OBJCLS_WEIGHT}
        actor = EXOTMergeActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, loss_type=cfg.MODEL.LOSS_TYPE)

    elif settings.script_name == "stark_lightning_X_trt":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKLightningXtrtActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")
    if settings.dry_run==1:
        w_cfg = None
    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    if settings.resume ==True:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True, load_previous_ckpt=True)
    elif settings.script_name in ["stark_st2", "stark_st2_plus_sp", "exot_st2"]:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True, load_previous_ckpt=True)
    else:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True)
    # TODO resume add
    if settings.dry_run==0:
        wandb.finish()
