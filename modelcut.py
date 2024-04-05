import os, torch, glob
from lib.models.exot import build_exotst, build_exotst1
net = build_exotst(cfg)

def save_checkpoint(self):
    """Saves a checkpoint of the network and other variables."""

    net =  self.actor.net

    actor_type = type(self.actor).__name__
    net_type = type(net).__name__
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
    }

    directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
    print(directory)
    if not os.path.exists(directory):
        print("directory doesn't exist. creating...")
        os.makedirs(directory)

    # First save as a tmp file
    tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
    torch.save(state, tmp_file_path)

    file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

    # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
    os.rename(tmp_file_path, file_path)


def load_state_dict( checkpoint=None, distill=False):
    """Loads a network checkpoint file.

    Can be called in three different ways:
        load_checkpoint():
            Loads the latest epoch from the workspace. Use this to continue training.
        load_checkpoint(epoch_num):
            Loads the network at the given epoch number (int).
        load_checkpoint(path_to_checkpoint):
            Loads the file from the given absolute path (str).
    """

    net = self.actor.net

    net_type = type(net).__name__

    if isinstance(checkpoint, str):
        # checkpoint is the path
        if os.path.isdir(checkpoint):
            checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                raise Exception('No checkpoint found')
        else:
            checkpoint_path = os.path.expanduser(checkpoint)
    else:
        raise TypeError

    # Load network
    print("Loading pretrained model from ", checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

    missing_k, unexpected_k = net.load_state_dict(checkpoint_dict["net"], strict=False)
    print("previous checkpoint is loaded.")
    print("missing keys: ", missing_k)
    print("unexpected keys:", unexpected_k)

    return True