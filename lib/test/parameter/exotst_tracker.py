from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.exot_st2.config import cfg, update_config_from_file


def parameters(modelname, yaml_name: str, checkpointname):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (modelname, yaml_name))
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE


    # Network checkpoint path #%s_fold_0_5
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/exot_st2/EXOTST_epoch=%d-v3-momentum.pth.tar" %
    #                                  ( 49)) #cfg.TEST.EPOCH))
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/%s/%s" %
                                     ( modelname, checkpointname)) #cfg.TEST.EPOCH))
    # lowdim
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
