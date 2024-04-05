import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import copy

class UR5Dataset(BaseDataset):
    def __init__(self):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = self.env_settings.robot_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/data_RGB/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        # with open(anno_path) as f:
        #     tmpgt = f.readline().split(',')[1:5]
        ground_truth_rect = load_text(str(anno_path), delimiter='\t', dtype=np.float64)
        #ground_truth_rect = np.array(tmpgt, dtype=np.float64)
        head = np.expand_dims(copy.deepcopy(ground_truth_rect[0]), 0)
        ground_truth_rect = np.concatenate([head, ground_truth_rect], axis=0)     

        frames_path = '{}/data_RGB/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".png")]
        frame_list.sort(key=lambda f: f[:-4])
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        # print(frames_list)

        gt, target_noun, objname = self.read_meta(frames_path, ground_truth_rect)

        return Sequence(sequence_name, frames_list, 'robot', ground_truth_rect.reshape(-1, 4), object_class=target_noun, target_visible=gt, obj_name=objname)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        print(self.base_path)
        with open('{}/data_RGB/test_seq2.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        return sequence_list

    def read_meta(self, frames_path, ground_truth_rect):
        obj_path = os.path.join(self.base_path, 'data_RGB', 'object_set.txt')
        with open(obj_path, 'r') as f:
            obj_list = f.read().splitlines()
        self.obj_dict = dict()
        for i in range(len(obj_list)):
            self.obj_dict[obj_list[i]] = 34+i
        objname = frames_path.split('/')[-1].split('-')[0]
        target_noun = self.obj_dict[objname]
        
        # gt = torch.unsqueeze(torch.any(gt<0, dim=-1).type(torch.uint8), dim=-1)
        # print(" AFTER shrink ", gt.shape)
        gt = np.any(ground_truth_rect>=0, axis=-1).astype(np.float)
        return gt, target_noun, objname
