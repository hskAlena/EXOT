import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import copy


class Trek150(BaseVideoDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().trek150_dir if root is None else root
        super().__init__('TREK150', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        noun_path = os.path.join(self.root, 'noun_class.csv')
        gtnoun = pandas.read_csv(noun_path, delimiter=',', na_filter=False).values
        self.noun_dict = dict()
        for i in range(len(gtnoun)):
            self.noun_dict[gtnoun[i][0]] = gtnoun[i][1]

        before_path = os.path.join(self.root, 'convert.csv')
        beforenoun = pandas.read_csv(before_path, delimiter=',', na_filter=False).values
        self.convert_dict = dict()
        for i in range(len(beforenoun)):
            self.convert_dict[beforenoun[i][1]] = beforenoun[i][0]

        # seq_id is the index of the folder inside the got10k root path
        seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'trek150'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False  # Has occlusion info globally, but not in locally

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            #print(seq_path)
            with open(os.path.join(seq_path, 'action_target.txt')) as f:
                meta_info = f.readlines()                  
            action_verb = int(meta_info[0][:-1])
            action_noun = int(meta_info[1][:-1])
            target_noun = int(meta_info[2][:-1])
            target_noun = int(self.convert_dict[target_noun])
            # target_noun = self.noun_dict[target_noun]

            with open(os.path.join(seq_path, 'anchors.txt')) as f:
                meta_infos = f.readlines()
            anchor_dict = {}
            for info in meta_infos:
                anchor_dict[int(info.split(',')[0])] = int(info.split(',')[1][:-1])

            with open(os.path.join(seq_path, 'attributes.txt')) as f:
                meta_infos = f.readlines()
            attribute_set = set()
            for meta in meta_infos:
                attribute_set.add(meta[:-1])

            with open(os.path.join(seq_path, 'frames.txt')) as f:
                meta_infos = f.readlines()
            firstframe = int(meta_infos[0][:-1])
        except:
            print("ERROR with meta info: No data")
            exit(0)   

        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        gt = torch.tensor(gt)

        gt = torch.any(gt<0, dim=-1).float()
        # gt = torch.unsqueeze(gt, 1)

        object_meta = OrderedDict({'object_class_name': target_noun,
                                       'motion_class': action_noun,
                                       'anchor_dict': anchor_dict,
                                       'attribute_set': attribute_set,
                                       'motion_adverb': action_verb,
                                       'first_frame': firstframe,
                                       'exit_flag': gt})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self, split):
        if split=='train':
            with open(os.path.join(self.root, 'sequences.txt')) as f:
                dir_list = list(csv.reader(f))
        elif split=='val':
            with open(os.path.join(self.root, 'whole_seq.txt')) as f:
                dir_list = list(csv.reader(f))
        elif split=='test':
            with open(os.path.join(self.root, 'test_seq2.txt')) as f:
                dir_list = list(csv.reader(f))

        dir_list = [dir_name[0].split('-')[0]+'/'+dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        gt = torch.tensor(gt)
        gt[:,0] = gt[:,0]/1920*456
        gt[:,1] = gt[:,1]/1080*256
        gt[:,2] = gt[:,2]/1920*456
        gt[:,3] = gt[:,3]/1080*256
        #print(gt)
        return gt

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        length = bbox.shape[0]

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        visible = torch.ByteTensor([1 for i in range(length)])
        visible_ratio = 7*visible.float()/8

        #visible, visible_ratio = self._read_target_visible(seq_path)
        visible = visible & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_id, seq_path, frame_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        firstframe = obj_meta['first_frame']
        return os.path.join(seq_path, 'img', 'frame_{:010}.jpg'.format(firstframe+frame_id))    # frames start from 1

    def _get_frame(self, seq_id, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_id, seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_id, seq_path, f_id) for f_id in frame_ids]
        obj_meta_copy = copy.copy(obj_meta)

        # obj_meta_copy['depth'] = [obj_meta['depth'][f_id] for i, f_id in enumerate(frame_ids)]
        obj_meta_copy['exit_flag'] = [obj_meta['exit_flag'][f_id] for i, f_id in enumerate(frame_ids)] 
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta_copy
