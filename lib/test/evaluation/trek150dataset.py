import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import pandas


class TREK150Dataset(BaseDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self,split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = self.env_settings.trek150_path
        self.sequence_list = self._get_sequence_list(split)

        before_path = os.path.join(self.base_path, 'convert.csv')
        beforenoun = pandas.read_csv(before_path, delimiter=',', na_filter=False).values
        self.convert_dict = dict()
        for i in range(len(beforenoun)):
            self.convert_dict[beforenoun[i][1]] = beforenoun[i][0]

        noun_path = os.path.join(self.base_path, 'noun_class.csv')
        gtnoun = pandas.read_csv(noun_path, delimiter=',', na_filter=False).values
        self.noun_dict = dict()
        for i in range(len(gtnoun)):
            self.noun_dict[gtnoun[i][0]] = gtnoun[i][1]

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s.split('-')[0]+'/'+s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = pandas.read_csv(anno_path, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        ground_truth_rect[:,0] = ground_truth_rect[:,0]/1920*456
        ground_truth_rect[:,1] = ground_truth_rect[:,1]/1080*256
        ground_truth_rect[:,2] = ground_truth_rect[:,2]/1920*456
        ground_truth_rect[:,3] = ground_truth_rect[:,3]/1080*256
        # with open(anno_path) as f:
        #     tmpgt = f.readline()[:-1].split(',')
        # #ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        # ground_truth_rect = np.array(tmpgt, dtype=np.float64)
        # ground_truth_rect[0] = ground_truth_rect[0]/1920*456
        # ground_truth_rect[1] = ground_truth_rect[1]/1080*256
        # ground_truth_rect[2] = ground_truth_rect[2]/1920*456
        # ground_truth_rect[3] = ground_truth_rect[3]/1080*256

        frame_path = '{}/{}/frames.txt'.format(self.base_path, sequence_name)

        with open(frame_path) as f:
            tmpgt = f.readline()[:-1]
        init_data = {int(tmpgt): dict()} 

        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)
        #print(os.listdir(frames_path))
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: f[:-4])
        #print(frame_list)
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        sequence_name = sequence_name[4:]

        gt, target_noun, objname = self.read_meta(frames_path, ground_truth_rect)
        # print(gt, target_noun, objname)

        return Sequence(sequence_name, frames_list, 'trek150', ground_truth_rect.reshape(-1, 4), object_class=target_noun, obj_name= objname, target_visible=gt)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        print(self.base_path)
        with open('{}/test_seq2.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        return sequence_list

    def read_meta(self, frames_path, ground_truth_rect):
        seq_path = '/'.join(frames_path.split('/')[:-1])
        with open(os.path.join(seq_path, 'action_target.txt')) as f:
            meta_info = f.readlines()                  
            target_noun = int(meta_info[2][:-1])
            objname = self.noun_dict[target_noun]
            target_noun = self.convert_dict[target_noun]
        
        # gt = torch.unsqueeze(torch.any(gt<0, dim=-1).type(torch.uint8), dim=-1)
        # print(" AFTER shrink ", gt.shape)
        # print((ground_truth_rect>=0).shape, )
        gt = np.any(ground_truth_rect>=0, axis=-1).astype(np.float)
        return gt, target_noun, objname
