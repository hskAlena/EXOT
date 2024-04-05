class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/hskim/projects/mfmot/STARK'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/hskim/projects/mfmot/STARK/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/hskim/projects/mfmot/STARK/pretrained_networks'
        self.lasot_dir = '/home/hskim/projects/mfmot/STARK/data/lasot'
        self.got10k_dir = '/home/hskim/projects/mfmot/STARK/data/got10k'
        self.lasot_lmdb_dir = '/home/hskim/projects/mfmot/STARK/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/hskim/projects/mfmot/STARK/data/got10k_lmdb'
        self.trek150_dir = '/home/hskim/projects/mfmot/STARK/data/TREK-150'
        self.trackingnet_dir = '/home/hskim/projects/mfmot/STARK/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/hskim/projects/mfmot/STARK/data/trackingnet_lmdb'
        self.coco_dir = '/home/hskim/projects/mfmot/STARK/data/coco'
        self.coco_lmdb_dir = '/home/hskim/projects/mfmot/STARK/data/coco_lmdb'
        self.robot_dir = '/home/hskim/projects/mfmot/STARK/data/robot-data'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/hskim/projects/mfmot/STARK/data/vid'
        self.imagenet_lmdb_dir = '/home/hskim/projects/mfmot/STARK/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
