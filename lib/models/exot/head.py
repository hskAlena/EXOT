import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.models.exot.backbone import FrozenBatchNorm2d
from lib.models.exot.repvgg import RepVGGBlock
# import time
import numpy as np


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float()

    def forward(self, x, return_dist=False, softmax=1):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=1):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        self.coord_x = self.coord_x.type_as(score_map)
        self.coord_y = self.coord_y.type_as(score_map)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax==1:
                return exp_x, exp_y, prob_vec
            elif softmax==2:
                return exp_x, exp_y, score_map
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

class Corner_GPU_Predictor(nn.Module):
    """ Corner Predictor module"""
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_GPU_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class Corner_Exit_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False, drop_ratio = 0.5, abs=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.abs = abs
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        self.exit_layer = nn.Sequential(
            nn.Linear(self.feat_sz*self.feat_sz, self.feat_sz),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            nn.Linear(self.feat_sz, self.feat_sz),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            nn.Linear(self.feat_sz, 1)
            )
                        
        self.vacant = torch.zeros(1,1,self.feat_sz, self.feat_sz,dtype=torch.float32).cuda()

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, template_anno=None, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        self.make_delta_coord(template_anno)

        if x.dim() == 5:
            n = x.shape[0]
            score_map_tl_list = []
            score_map_br_list = []
            for i in range(n):
                score_map_tl, score_map_br = self.get_score_map(x[i])
                score_map_br_list.append(score_map_br)
                score_map_tl_list.append(score_map_tl)
            score_map_tl = torch.stack(score_map_tl_list, dim=0)
            score_map_br = torch.stack(score_map_br_list, dim=0)
        else:
            score_map_tl, score_map_br = self.get_score_map(x)

        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl, exit_t = self.soft_argmax(score_map_tl, 'tl', return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br, exit_b = self.soft_argmax(score_map_br, 'br', return_dist=True, softmax=softmax)
            if coorx_br.dim() == 2:
                return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=2) / self.img_sz, prob_vec_tl, prob_vec_br, [ exit_t, exit_b],
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br, [exit_t, exit_b],
        else:
            coorx_tl, coory_tl, exit_t = self.soft_argmax(score_map_tl, 'tl')
            coorx_br, coory_br, exit_b = self.soft_argmax(score_map_br, 'br')
            if coorx_br.dim() == 2:
                return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=2) / self.img_sz, [ exit_t, exit_b],
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, [ exit_t, exit_b],

    def make_delta_coord(self, template_anno):
        '''about coordinates and indexs'''
        with torch.no_grad():            
            tn, tb, _ = template_anno.shape
            index = torch.round(template_anno[0]*self.feat_sz*self.stride).int()
            # generate mesh-grid
            self.tl_coord_x = self.indice.repeat((self.feat_sz, 1)).repeat((tb, 1, 1)) \
                .view((tb, self.feat_sz * self.feat_sz,)).float().cuda() - torch.unsqueeze(index[:,0], 1).repeat((1, self.feat_sz*self.feat_sz)) 
            self.tl_coord_y = self.indice.repeat((1, self.feat_sz)).repeat((tb, 1, 1)) \
                .view((tb, self.feat_sz * self.feat_sz,)).float().cuda() - torch.unsqueeze(index[:,1], 1).repeat((1, self.feat_sz*self.feat_sz)) 
            # generate mesh-grid
            self.br_coord_x = self.indice.repeat((self.feat_sz, 1)).repeat((tb, 1, 1)) \
                .view((tb, self.feat_sz * self.feat_sz,)).float().cuda() - torch.unsqueeze(index[:,2], 1).repeat((1, self.feat_sz*self.feat_sz)) 
            self.br_coord_y = self.indice.repeat((1, self.feat_sz)).repeat((tb, 1, 1)) \
                .view((tb, self.feat_sz * self.feat_sz,)).float().cuda() - torch.unsqueeze(index[:,3], 1).repeat((1, self.feat_sz*self.feat_sz)) 

    
    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, tl_br_flg, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        if score_map.dim()==5:
            n, b, _, _, _ = score_map.shape
            score_vec = score_map.view((n, b, self.feat_sz*self.feat_sz))
            prob_vec = nn.functional.softmax(score_vec, dim=2)
            exit_logit = self.exit_layer(score_vec)
        else:
            score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
            exit_logit = self.exit_layer(score_vec)
            prob_vec = nn.functional.softmax(score_vec, dim=1)

        if self.abs:
            coord_x = self.coord_x
            coord_y = self.coord_y
        else:
            if tl_br_flg == 'tl':
                coord_x = self.tl_coord_x
                coord_y = self.tl_coord_y
            else:
                coord_x = self.br_coord_x
                coord_y = self.br_coord_y

        if score_map.dim()==5:
            exp_x = torch.sum((coord_x * prob_vec), dim=2)
            exp_y = torch.sum((coord_y * prob_vec), dim=2) # average
        else:
            exp_x = torch.sum((coord_x * prob_vec), dim=1)
            exp_y = torch.sum((coord_y * prob_vec), dim=1) # average
        
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec, exit_logit
            else:
                return exp_x, exp_y, score_vec, exit_logit
        else:
            return exp_x, exp_y, exit_logit


class Corner_Predictor_Lite(nn.Module):
    """ Corner Predictor module (Lite version)"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16):
        super(Corner_Predictor_Lite, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''convolution tower for two corners'''
        self.conv_tower = nn.Sequential(conv(inplanes, channel),
                                        conv(channel, channel // 2),
                                        conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        nn.Conv2d(channel // 8, 2, kernel_size=3, padding=1))

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = (torch.arange(0, self.feat_sz).view(-1, 1) + 0.5) * self.stride  # here we can add a 0.5
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        score_map = self.conv_tower(x)  # (B,2,H,W)
        return score_map[:, 0, :, :], score_map[:, 1, :, :]

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class Corner_Predictor_Lite_Rep(nn.Module):
    """ Corner Predictor module (Lite version with repvgg style)"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16):
        super(Corner_Predictor_Lite_Rep, self).__init__()
        self.feat_sz = feat_sz
        self.feat_len = feat_sz ** 2
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''convolution tower for two corners'''
        self.conv_tower = nn.Sequential(RepVGGBlock(inplanes, channel, kernel_size=3, padding=1),
                                        RepVGGBlock(channel, channel // 2, kernel_size=3, padding=1),
                                        RepVGGBlock(channel // 2, channel // 4, kernel_size=3, padding=1),
                                        RepVGGBlock(channel // 4, channel // 8, kernel_size=3, padding=1),
                                        nn.Conv2d(channel // 8, 2, kernel_size=3, padding=1))

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = (torch.arange(0, self.feat_sz).view(-1, 1) + 0.5) * self.stride  # here we can add a 0.5
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        # s = time.time()
        score_map_tl, score_map_br = self.get_score_map(x)
        # e1 = time.time()
        # print("head forward time: %.2f ms" % ((e1-s)*1000))
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            # e2 = time.time()
            # print("soft-argmax time: %.2f ms" % ((e2 - e1) * 1000))
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        score_map = self.conv_tower(x)  # (B,2,H,W)
        return score_map[:, 0, :, :], score_map[:, 1, :, :]

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_len))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class Corner_Predictor_Lite_Rep_v2(nn.Module):
    """ Corner Predictor module (Lite version with repvgg style)"""

    def __init__(self, inplanes=128, channel=128, feat_sz=20, stride=16):
        super(Corner_Predictor_Lite_Rep_v2, self).__init__()
        self.feat_sz = feat_sz
        self.feat_len = feat_sz ** 2
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''convolution tower for two corners'''
        self.conv_tower = nn.Sequential(RepVGGBlock(inplanes, channel, kernel_size=3, padding=1),
                                        RepVGGBlock(channel, channel, kernel_size=3, padding=1),
                                        nn.Conv2d(channel, 2, kernel_size=3, padding=1))

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = (torch.arange(0, self.feat_sz).view(-1, 1) + 0.5) * self.stride  # here we can add a 0.5
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        # s = time.time()
        score_map_tl, score_map_br = self.get_score_map(x)
        # e1 = time.time()
        # print("head forward time: %.2f ms" % ((e1-s)*1000))
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            # e2 = time.time()
            # print("soft-argmax time: %.2f ms" % ((e2 - e1) * 1000))
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        score_map = self.conv_tower(x)  # (B,2,H,W)
        return score_map[:, 0, :, :], score_map[:, 1, :, :]

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_len))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ODIN_large_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        self.BN = BN
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            ## to do batchnorm [16, 1, 256] and batchnorm 1 ?
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h[:-1], h))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h[:-1], h))


        if type(output_dim) == list:
            output_dim = np.sum(np.array(output_dim))
        self.h = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_normal_(self.h.weight, nonlinearity='relu')
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

        self.g = nn.Sequential(nn.Linear(hidden_dim, 1), nn.BatchNorm1d(1))
        # self.g = nn.Linear(hidden_dim, 1)
        # kernel regularizer by SGD optimizer

    def forward(self, x):
        # print(x.shape, "ODIN shape")
        # torch.Size([1, 16, 1, 256]) ODIN shape
        if self.BN==True:
            pass
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        h_feature = self.h(x)
        g_feature = torch.sigmoid(self.g(x))
        x = torch.div(h_feature, g_feature)

        return x, h_feature, g_feature

class ODIN_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        self.BN = BN
        h = [hidden_dim] * (num_layers - 1)
        self.reduceNet = nn.Sequential(nn.Unflatten(2, (20,20)),
                                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if BN:
            ## to do batchnorm [16, 1, 256] and batchnorm 1 ?            
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h[:-1], h))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h[:-1], h))

        if type(output_dim) == list:
            output_dim = np.sum(np.array(output_dim))
        self.h = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_normal_(self.h.weight, nonlinearity='relu')
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

        self.g = nn.Sequential(nn.Linear(hidden_dim, 1), nn.BatchNorm1d(1))
        # self.g = nn.Linear(hidden_dim, 1)
        # kernel regularizer by SGD optimizer

    def forward(self, x):
        # print(x.shape, "ODIN shape")
        # torch.Size([1, 16, 1, 256]) ODIN shape
        if self.BN==True:
            pass
        x = self.reduceNet(x)
        x = torch.squeeze(self.avgpool(x))
        # if batch_size == 1
        if x.dim()<2:
            x = torch.unsqueeze(x, 0)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        h_feature = self.h(x)
        g_feature = torch.sigmoid(self.g(x))
        x = torch.div(h_feature, g_feature)

        return x, h_feature, g_feature


def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

class ODIN_COS_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        self.BN = BN
        h = [hidden_dim] * (num_layers - 1)
        self.reduceNet = nn.Sequential(nn.Unflatten(2, (20,20)),
                                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if BN:
            ## to do batchnorm [16, 1, 256] and batchnorm 1 ?
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h[:-1], h))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h[:-1], h))


        if type(output_dim) == list:
            output_dim = np.sum(np.array(output_dim))
        self.h = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_normal_(self.h.weight, nonlinearity='relu')
        self.g = nn.Sequential(nn.Linear(hidden_dim, 1), nn.BatchNorm1d(1))
        # self.g = nn.Linear(hidden_dim, 1)
        # kernel regularizer by SGD optimizer

    def forward(self, x):
        # print(x.shape, "ODIN shape")
        # torch.Size([1, 16, 1, 256]) ODIN shape
        if self.BN==True:
            pass
        x = self.reduceNet(x)
        x = torch.squeeze(self.avgpool(x))
        # if batch_size == 1
        if x.dim()<2:
            x = torch.unsqueeze(x, 0)

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        
        x_norm = norm(x)
        w = norm(self.h.weight)
        h_feature = torch.matmul(x_norm, w.T)
        g_feature = torch.sigmoid(self.g(x))
        x = torch.div(h_feature, g_feature)

        return x, h_feature, g_feature

class CLS_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        self.BN = BN
        h = [hidden_dim] * (num_layers - 1)
        if type(output_dim) == list:
            output_dim = np.sum(np.array(output_dim))
        self.reduceNet = nn.Sequential(nn.Unflatten(2, (20,20)),
                                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if BN:
            ## to do batchnorm [16, 1, 256] and batchnorm 1 ?
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))       

    def forward(self, x):
        x = self.reduceNet(x)
        x = torch.squeeze(self.avgpool(x))
        # if batch_size == 1
        if x.dim()<2:
            x = torch.unsqueeze(x, 0)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ODIN_EUC_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        self.BN = BN
        h = [hidden_dim] * (num_layers - 1)
        self.reduceNet = nn.Sequential(nn.Unflatten(2, (20,20)),
                                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if BN:
            ## to do batchnorm [16, 1, 256] and batchnorm 1 ?
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h[:-1], h))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h[:-1], h))


        if type(output_dim) == list:
            output_dim = np.sum(np.array(output_dim))
        self.h = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_normal_(self.h.weight, nonlinearity='relu')
        self.g = nn.Sequential(nn.Linear(hidden_dim, 1), nn.BatchNorm1d(1))
        # self.g = nn.Linear(hidden_dim, 1)
        # kernel regularizer by SGD optimizer

    def forward(self, x):
        # print(x.shape, "ODIN shape")
        # torch.Size([1, 16, 1, 256]) ODIN shape
        if self.BN==True:
            pass
        x = self.reduceNet(x)
        x = torch.squeeze(self.avgpool(x))
        # if batch_size == 1
        if x.dim()<2:
            x = torch.unsqueeze(x, 0)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        
        x_norm = x.unsqueeze(2) #(batch, latent, 1)
        h = self.h.weight.T.unsqueeze(0) #(1, latent, num_classes)
        h_feature = -((x_norm -h).pow(2)).mean(1)
        g_feature = torch.sigmoid(self.g(x))
        x = torch.div(h_feature, g_feature)

        return x, h_feature, g_feature


def build_box_head(cfg):
    if cfg.MODEL.HEAD_TYPE == "MLP":
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif "CORNER" in cfg.MODEL.HEAD_TYPE:
        if cfg.MODEL.BACKBONE.DILATION is False:
            stride = 16
        else:
            stride = 8
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "HEAD_DIM", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD_TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        elif cfg.MODEL.HEAD_TYPE == "CORNER_EXIT":
            corner_head = Corner_Exit_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride, abs = cfg.MODEL.HEAD_ABS)
        elif cfg.MODEL.HEAD_TYPE == "CORNER_LITE":
            corner_head = Corner_Predictor_Lite(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                                feat_sz=feat_sz, stride=stride)
        elif cfg.MODEL.HEAD_TYPE == "CORNER_LITE_REP":
            corner_head = Corner_Predictor_Lite_Rep(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                                    feat_sz=feat_sz, stride=stride)
        elif cfg.MODEL.HEAD_TYPE == "CORNER_LITE_REP_v2":
            corner_head = Corner_Predictor_Lite_Rep_v2(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                                       feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)
