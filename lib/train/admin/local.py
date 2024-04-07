class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/hskim/projects/EXOT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/hskim/projects/EXOT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/hskim/projects/EXOT/pretrained_networks'
        self.trek150_dir = '/home/hskim/projects/EXOT/data/TREK-150'
        self.robot_dir = '/home/hskim/projects/EXOT/data/robot-data'
