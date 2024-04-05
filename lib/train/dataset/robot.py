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
import pickle as pkl
import copy


class UR5VideoDataset(BaseVideoDataset):
    def __init__(self, root=None, interpolation=False, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
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
        root = env_settings().robot_dir if root is None else root
        super().__init__('ROBOT', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        # seq_id is the index of the folder inside the got10k root path
        seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()
        self.interpolation = interpolation

        # self.joint_path = os.path.join(self.root, 'data_getl', 'auto')
        # self.getj_getl = 'getl'

    def get_name(self):
        return 'robot'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False  # Has occlusion info globally, but not in locally

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(s) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        #self.depth_path = os.path.join(self.root, 'data_Depth', seq_path)
        #self.joint_path = os.path.join(self.root, 'data_getl', 'auto')
        obj_path = os.path.join(self.root, 'data_RGB', 'object_set.txt')
        with open(obj_path, 'r') as f:
            obj_list = f.read().splitlines()
        self.obj_dict = dict()
        for i in range(len(obj_list)):
            self.obj_dict[obj_list[i]] = i #34+i

        rgb_path = os.path.join(self.root, 'data_RGB', seq_path)
        bb_anno_file = os.path.join(rgb_path, "groundtruth.txt")
        

        gt = pandas.read_csv(bb_anno_file, delimiter='\t', header= None, dtype=np.float32, na_filter=False, low_memory=False).values
        gt = torch.tensor(gt)

        # gt = torch.unsqueeze(torch.any(gt<0, dim=-1).type(torch.uint8), dim=-1)
        # print(" AFTER shrink ", gt.shape)
        gt = torch.any(gt<0, dim=-1).float()
        # gt = torch.unsqueeze(gt, 1)

        # if os.path.isdir(self.joint_path):
        #     self.getj_getl = 'getl'
        #     self.joint_path = os.path.join(self.root, 'data_getl', seq_path)
        # else:
        #     self.getj_getl = 'getj'
        #     self.joint_path = os.path.join(self.root, 'data_getj', seq_path)

        meta_anno_file = os.path.join(self.root, "data_RGB", seq_path, "groundmeta.txt")
        if os.path.isfile(meta_anno_file):
            with open(meta_anno_file, 'r') as f:
                metaline = f.readline()
            target_noun = self.obj_dict[metaline.split(',')[0]]
            firstframe = metaline.split(',')[5]
        else:
            target_noun = self.obj_dict[seq_path.split('/')[-1].split('-')[0]]
            firstframe = seq_path.split('/')[-1]+'_frame_00001.png'

        # with open(self.joint_path+'.pkl', 'rb') as f:
        #     metajoint = pkl.load(f)  #list
        # metajoint = torch.tensor(np.array(metajoint)) #[Len, 6]

        # if gt.shape[0] != metajoint.shape[0]:
        #     # print(gt.shape, metajoint.shape)
        #     head = torch.unsqueeze(copy.deepcopy(gt[0]), 0)
        #     gt = torch.cat([head, gt], dim=0)

        # with open(self.depth_path+'.pkl', 'rb') as f:
        #     metadepth = pkl.load(f, encoding="latin1")  #list max=1490     
        # metadepth = torch.Tensor(np.int_(np.array(metadepth))) #numpy.array (480, 640) dtype=uint16 -> int32

        # object_meta = OrderedDict({'object_class_name': target_noun,
        #                                'first_frame': firstframe,
        #                                'exit_flag': gt,
        #                                'depth': metadepth,
        #                                'joint': metajoint,
        #                                'joint_flag': self.getj_getl})

        object_meta = OrderedDict({'object_class_name': target_noun,
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
            with open(os.path.join(self.root, 'data_RGB/sequence_list.txt')) as f:
                dir_list = list(csv.reader(f))
        elif split=='val':
            with open(os.path.join(self.root, 'data_RGB/whole_seq.txt')) as f:
                dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def fix_bbox(self, bbox):
        if np.count_nonzero(bbox<0)==0:
            return bbox

        xp = np.unique(np.nonzero(bbox<0)[0])
        total = []
        sub = []
        for i in range(len(xp)-1):
            if len(sub)==0:
                sub.append(xp[i])
            if xp[i+1] == xp[i]+1:
                sub.append(xp[i+1])
            else:
                total.append(sub)
                sub = []
                # print('reset', i)
            if i== len(xp)-2:
                total.append(sub)

        final = bbox.shape[0]
        def y_idx(idx):
            for i in range(len(total)):
                startp = total[i][0] - 7
                xp = np.arange(startp, total[i][0])
                yp = bbox[xp,idx]
                if idx == 0 or idx ==2:
                    thre_max = 540
                    maxnum = 640
                elif idx == 1 or idx ==3:
                    thre_max = 380
                    maxnum = 480

                if np.max(yp)>thre_max:
                    thres = maxnum
                elif np.min(yp) <50:
                    thres= 0
                elif np.mean(yp)>int(thre_max/2):
                    thres = np.max(yp)
                else:
                    thres = np.min(yp)
                    
                # if total[i][-1]==final-1:
                result = np.interp(total[i], xp, yp, right=thres)
            
                # print(gt[startp:total[i][-1], 0])
                bbox[total[i], idx] = result
                # print(gt[startp:total[i][-1], 0])
                # print("CONVENT", len(total[i]), len(result))
        y_idx(0)
        y_idx(1)
        y_idx(2)
        y_idx(3)
        return bbox

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter='\t', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        if self.interpolation:
            gt = self.fix_bbox(gt)
        gt = torch.tensor(gt)

        head = torch.unsqueeze(copy.deepcopy(gt[0]), 0)
        gt = torch.cat([head, gt], dim=0)
        # print(gt.shape)
        return gt

    def _get_sequence_path(self, seq_id):        
        return os.path.join(self.root, 'data_RGB', self.sequence_list[seq_id])

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
        # print("GET frame path", seq_id, frame_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        firstframe = obj_meta['first_frame']
        return os.path.join(seq_path, firstframe.split('_')[0]+'_frame_{:05}.png'.format(1+frame_id))    # frames start from 1

    def _get_frame(self, seq_id, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_id, seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):

        seq_path = self._get_sequence_path(seq_id)
        #print("ONE get frames", seq_id, frame_ids)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        obj_meta_copy = copy.copy(obj_meta)

        # obj_meta_copy['depth'] = [obj_meta['depth'][f_id] for i, f_id in enumerate(frame_ids)]
        obj_meta_copy['exit_flag'] = [obj_meta['exit_flag'][f_id] for i, f_id in enumerate(frame_ids)]        
        # obj_meta_copy['joint'] = [obj_meta['joint'][f_id] for i, f_id in enumerate(frame_ids)]
        frame_list = [self._get_frame(seq_id, seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]


        #print(len(frame_list), anno_frames)
        return frame_list, anno_frames, obj_meta_copy
