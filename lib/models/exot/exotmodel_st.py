"""
STARK-ST Model (Spatio-Temporal).
"""
from .backbone import build_backbone
from .transformer import build_transformer
from .head import build_box_head, ODIN_large_MLP, ODIN_MLP, MLP, ODIN_COS_MLP, ODIN_EUC_MLP, CLS_MLP
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.misc import NestedTensor
import torch
from torch import nn
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.merge import merge_template_search


class EXOTST(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type="CORNER", cls_head=None, obj_cls = None, abs_type=False, cls_type='ori'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries
        self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        self.aux_loss = aux_loss
        self.head_type = head_type
        self.abs = abs_type
        if head_type == "CORNER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.cls_head = cls_head
        self.obj_cls = obj_cls
        self.cls_type = cls_type

    def forward(self, img=None, seq_dict=None, annot = None, mode="backbone", run_box_head=False, run_cls_head=False):
        if mode == "backbone":
            return self.forward_backbone(img)
        elif mode == "transformer":
            return self.forward_transformer(seq_dict, annot = annot, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(output_back, pos)

    def forward_transformer(self, seq_dict, annot = None, run_box_head=False, run_cls_head=False, batch_feat=None):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        output_embed, enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], self.query_embed.weight,
                                                 seq_dict["pos"], return_encoder_output=True)
        # Forward the corner head
        out, outputs_coord, batch_feat = self.forward_head(output_embed, enc_mem, annot = annot, seq_dict = seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head, batch_feat=batch_feat)
        return out, batch_feat, output_embed, enc_mem

    def forward_box_head(self, hs, memory, annot = None):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER_EXIT":
            # adjust shape
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            template_anno, template_joint = annot
            tmpbox, prob_vec_tl, prob_vec_br, flag = self.box_head(opt_feat, template_anno, return_dist=True, softmax=False)
            if tmpbox.dim()==5:
                infn = tmpbox.shape[0]
                tmplate_anot = template_anno[0].repeat(infn,1,1)
            else:
                tmplate_anot = template_anno[0]
            if not self.abs:
                tmpbox = tmplate_anot+ tmpbox
            # print('template', template_joint)
            if not self.abs and template_joint!= None:
                # print(tmpbox.shape) # (2, 8, 4)
                if tmpbox.dim()==3:
                    tmpbox_xmean = torch.unsqueeze((tmpbox[:,:,0]+tmpbox[:,:,2])/200, -1)
                    tmpbox_ymean = torch.unsqueeze((tmpbox[:,:,1]+tmpbox[:,:,3])/200, -1)
                    resjoint = torch.unsqueeze(template_joint[0, :, :2], 0)+torch.cat([tmpbox_xmean, tmpbox_ymean], -1)
                    resjoint = resjoint.view(-1,2)
                else:
                    tmpbox_xmean = torch.unsqueeze((tmpbox[:,0]+tmpbox[:,2])/200, 1)
                    tmpbox_ymean = torch.unsqueeze((tmpbox[:,1]+tmpbox[:,3])/200, 1)
                    resjoint = template_joint[0, :, :2]+torch.cat([tmpbox_xmean, tmpbox_ymean], 1)
            else:
                resjoint = None

            outputs_coord = box_xyxy_to_cxcywh(tmpbox)
            # print(opt_feat.shape, "FEATRE SHAOE")  # [16 256 20 20] 
            if outputs_coord.dim() !=3:
                b, _, _, _ = opt_feat.shape
                outputs_coord_new = outputs_coord.view(b, 1, 4)
            else:
                n, b, _, _, _ = opt_feat.shape
                outputs_coord_new = outputs_coord.view(b, n, 4)
            flag = [prob_vec_tl, prob_vec_br, flag]

            if template_joint !=None:
                out = {'pred_boxes': outputs_coord_new, 'pred_joint': resjoint}
            else:
                out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord, (flag, opt_feat), 

            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            # out = {'pred_boxes': outputs_coord_new}
            # return out, outputs_coord_new
        elif self.head_type == "MLP":
            # Forward the class and box head
            outputs_coord = self.box_head(hs).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord, None, 
        elif self.head_type=='CORNER':
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new, None,

    
    def forward_head(self, hs, memory, annot = None, seq_dict = None, run_box_head=False, run_cls_head=False, batch_feat=None):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        out_dict = {}
        if run_cls_head:
            # forward the classification head
            torch.set_printoptions(threshold=30_000)
            # print(seq_dict['feat_x'].shape, seq_dict['mask_x'].shape) #torch.Size([400, 16, 256]) torch.Size([1, 16, 1, 256])
            # print(seq_dict['mask_x']) # 16, 400
            # 320 , stride 16 -> 20 , 20*20 -> HIDDEN_DIM
            # torch.Size([16, 400]) torch.Size([16, 400]) torch.Size([16, 102400])

            if batch_feat ==None:
                if self.abs == 'backbone':
                    batch_feat = seq_dict['feat_x'].permute((1, 0, 2))            
                elif self.abs == 'enc_feat':
                    batch_feat = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
                elif self.abs == 'sim_feat':
                    enc_opt = memory[-self.feat_len_s:].transpose(0, 1)
                    dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
                    att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
                    opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)) #.permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
                    batch_feat = nn.functional.normalize(torch.squeeze(opt, -1), p=1.0)
                elif self.abs == 'hs':
                    batch_feat = torch.squeeze(hs)            
            batch_feat = batch_feat.permute(0,2,1)
            if self.cls_type == 'cls':
                tmp_obj = self.obj_cls(batch_feat)
                out_dict.update({'pred_logits': self.cls_head(hs)[-1], 'pred_obj': tmp_obj })
            else:
                tmp_obj, h, g = self.obj_cls(batch_feat)

                # print("AFTER ", torch.min(tmp_obj[0]), torch.max(tmp_obj[0]))

                # print(torch.min(hs[0][0]), torch.max(hs[0][0]))
                # print(tmp_obj.shape) #torch.Size([16, 366])
                #torch.Size([1, 16, 1, 366]) TMP hs
                out_dict.update({'pred_logits': self.cls_head(hs)[-1], 'pred_obj': tmp_obj, 'pred_h': h, 'pred_g': g})
            # forward the box prediction head
        if run_box_head:
            # forward the box prediction head
            out_dict_box, outputs_coord, flagFeat = self.forward_box_head(hs, memory, annot = annot)
            # merge results
            out_dict.update(out_dict_box)
            return out_dict, outputs_coord, batch_feat
        else:
            return out_dict, None, None, 

    def adjust(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]

def build_exotst_odin(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)
    if cfg.MODEL.HEAD_ABS == 'hs':
        if cfg.MODEL.ODIN_TYPE == 'ori':
            odin_cls = ODIN_MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, cfg.DATA.TRAIN.DATASETS_OBJNUM, cfg.MODEL.NLAYER_HEAD)
        elif cfg.MODEL.ODIN_TYPE == 'cos':
            odin_cls = ODIN_COS_MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, cfg.DATA.TRAIN.DATASETS_OBJNUM, cfg.MODEL.NLAYER_HEAD)
        elif cfg.MODEL.ODIN_TYPE == 'euc':
            odin_cls = ODIN_EUC_MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, cfg.DATA.TRAIN.DATASETS_OBJNUM, cfg.MODEL.NLAYER_HEAD)
    else:
        if cfg.MODEL.ODIN_TYPE == 'ori':
            odin_cls = ODIN_MLP(2*cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, cfg.DATA.TRAIN.DATASETS_OBJNUM, cfg.MODEL.NLAYER_HEAD)
        elif cfg.MODEL.ODIN_TYPE == 'cos':
            odin_cls = ODIN_COS_MLP(2*cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, cfg.DATA.TRAIN.DATASETS_OBJNUM, cfg.MODEL.NLAYER_HEAD)
        elif cfg.MODEL.ODIN_TYPE == 'euc':
            odin_cls = ODIN_EUC_MLP(2*cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, cfg.DATA.TRAIN.DATASETS_OBJNUM, cfg.MODEL.NLAYER_HEAD)
    
    cls_head = MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, 1, cfg.MODEL.NLAYER_HEAD)
    model = EXOTST1(
        backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        cls_head=cls_head,
        odin_cls = odin_cls,
        abs_type = cfg.MODEL.HEAD_ABS,
        cls_type = cfg.MODEL.ODIN_TYPE
    )

    return model

def build_exotst_cls(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)
    if cfg.MODEL.HEAD_ABS == 'hs':        
        obj_cls = CLS_MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, cfg.DATA.TRAIN.DATASETS_OBJNUM, cfg.MODEL.NLAYER_HEAD)
    else:
        obj_cls = CLS_MLP(2*cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, cfg.DATA.TRAIN.DATASETS_OBJNUM, cfg.MODEL.NLAYER_HEAD)
            
    cls_head = MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, 1, cfg.MODEL.NLAYER_HEAD)
    model = EXOTST(
        backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        cls_head=cls_head,
        obj_cls = obj_cls,
        abs_type = cfg.MODEL.HEAD_ABS,
        cls_type = cfg.MODEL.ODIN_TYPE
    )

    return model

class EXOTST1(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type="CORNER", cls_head=None, odin_cls=None, abs_type=False, cls_type='ori'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries
        self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        self.aux_loss = aux_loss
        self.head_type = head_type
        self.abs = abs_type
        if head_type == "CORNER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.cls_head = cls_head
        self.odin_cls = odin_cls
        self.cls_type = cls_type

    def forward(self, img=None, seq_dict=None, annot = None, mode="backbone", run_box_head=False, run_cls_head=False):
        if mode == "backbone":
            return self.forward_backbone(img)
        elif mode == "transformer":
            return self.forward_transformer(seq_dict, annot = annot, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(output_back, pos)

    def forward_transformer(self, seq_dict, annot = None, run_box_head=False, run_cls_head=False, batch_feat=None):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        output_embed, enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], self.query_embed.weight,
                                                 seq_dict["pos"], return_encoder_output=True)
        # Forward the corner head
        out, outputs_coord, batch_feat = self.forward_head(output_embed, enc_mem, annot = annot, seq_dict = seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head, batch_feat=batch_feat)
        return out, batch_feat, output_embed, enc_mem
    
    def forward_CAM_single(self, data, output_type='pred_obj'):
        #output_type = ['pred_logits', 'pred_obj', 'pred_h', 'pred_g', 'pred_boxes']
        out_dict = dict()
        feat_dict_list = []
        # process the templates
        for i in range(2):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            feat_dict_list.append(self.forward_backbone(NestedTensor(template_img_i, template_att_i)))

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        feat_dict_list.append(self.forward_backbone(NestedTensor(search_img, search_att)))

        # run the transformer and compute losses
        seq_dict = merge_template_search(feat_dict_list, return_search=True, return_template=True)
        
        template_bboxes = box_xywh_to_xyxy(data['template_anno'])  #(N_t, batch, 4)
        template_joint = None
        annot = (template_bboxes, template_joint) 

        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        output_embed, enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], self.query_embed.weight,
                                                 seq_dict["pos"], return_encoder_output=True)

        if self.abs == 'backbone':
            batch_feat = seq_dict['feat_x'].permute((1, 0, 2))            
        elif self.abs == 'enc_feat':
            batch_feat = enc_mem[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
        elif self.abs == 'sim_feat':
            enc_opt = enc_mem[-self.feat_len_s:].transpose(0, 1)
            dec_opt = output_embed.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)) #.permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            # print("opt shape", opt.shape)
            batch_feat = nn.functional.normalize(torch.squeeze(opt, -1), p=1.0)
        elif self.abs == 'hs':
            batch_feat = torch.squeeze(output_embed)     
            
        if batch_feat.dim() <2:
            batch_feat = torch.unsqueeze(batch_feat, 0)

        batch_feat = batch_feat.permute(0,2,1)
        if self.cls_type == 'cls':
            tmp_obj = self.obj_cls(batch_feat)
            out_dict.update({'pred_logits': self.cls_head(output_embed)[-1], 'pred_obj': tmp_obj })
        else:
            tmp_obj, h, g = self.odin_cls(batch_feat)

            out_dict.update({'pred_logits': self.cls_head(output_embed)[-1], 'pred_obj': tmp_obj, 'pred_h':h, 'pred_g':g})
            # forward the box prediction head
        out_dict_box, outputs_coord, flagFeat = self.forward_box_head(output_embed, enc_mem, annot = annot)
        # merge results
        out_dict.update(out_dict_box)

        if output_type == 'pred_logits':
            return out_dict['pred_logits']
        elif output_type == 'pred_obj':
            return out_dict['pred_obj']
        elif output_type == 'pred_h':
            return out_dict['pred_h']
        elif output_type == 'pred_boxes':
            return out_dict['pred_boxes']

    def forward_box_head(self, hs, memory, annot = None):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER_EXIT":
            # adjust shape
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            template_anno, template_joint = annot
            tmpbox, prob_vec_tl, prob_vec_br, flag = self.box_head(opt_feat, template_anno, return_dist=True, softmax=False)
            if tmpbox.dim()==5:
                infn = tmpbox.shape[0]
                tmplate_anot = template_anno[0].repeat(infn,1,1)
            else:
                tmplate_anot = template_anno[0]
            if not self.abs:
                tmpbox = tmplate_anot+ tmpbox
            # print('template', template_joint)
            if not self.abs and template_joint!= None:
                # print(tmpbox.shape) # (2, 8, 4)
                if tmpbox.dim()==3:
                    tmpbox_xmean = torch.unsqueeze((tmpbox[:,:,0]+tmpbox[:,:,2])/200, -1)
                    tmpbox_ymean = torch.unsqueeze((tmpbox[:,:,1]+tmpbox[:,:,3])/200, -1)
                    resjoint = torch.unsqueeze(template_joint[0, :, :2], 0)+torch.cat([tmpbox_xmean, tmpbox_ymean], -1)
                    resjoint = resjoint.view(-1,2)
                else:
                    tmpbox_xmean = torch.unsqueeze((tmpbox[:,0]+tmpbox[:,2])/200, 1)
                    tmpbox_ymean = torch.unsqueeze((tmpbox[:,1]+tmpbox[:,3])/200, 1)
                    resjoint = template_joint[0, :, :2]+torch.cat([tmpbox_xmean, tmpbox_ymean], 1)
            else:
                resjoint = None

            outputs_coord = box_xyxy_to_cxcywh(tmpbox)
            # print(opt_feat.shape, "FEATRE SHAOE")  # [16 256 20 20] 
            if outputs_coord.dim() !=3:
                b, _, _, _ = opt_feat.shape
                outputs_coord_new = outputs_coord.view(b, 1, 4)
            else:
                n, b, _, _, _ = opt_feat.shape
                outputs_coord_new = outputs_coord.view(b, n, 4)
            flag = [prob_vec_tl, prob_vec_br, flag]

            if template_joint !=None:
                out = {'pred_boxes': outputs_coord_new, 'pred_joint': resjoint}
            else:
                out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord, (flag, opt_feat), 

            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            # out = {'pred_boxes': outputs_coord_new}
            # return out, outputs_coord_new
        elif self.head_type == "MLP":
            # Forward the class and box head
            outputs_coord = self.box_head(hs).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord, None, 
        elif self.head_type=='CORNER':
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new, None,

    
    def forward_head(self, hs, memory, annot = None, seq_dict = None, run_box_head=False, run_cls_head=False, batch_feat=None):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        out_dict = {}
        if run_cls_head:
            # forward the classification head
            torch.set_printoptions(threshold=30_000)
            # print(seq_dict['feat_x'].shape, seq_dict['mask_x'].shape) #torch.Size([400, 16, 256]) torch.Size([1, 16, 1, 256])
            # print(seq_dict['mask_x']) # 16, 400
            # 320 , stride 16 -> 20 , 20*20 -> HIDDEN_DIM
            # torch.Size([16, 400]) torch.Size([16, 400]) torch.Size([16, 102400])

            if batch_feat==None:
                if self.abs == 'backbone':
                    batch_feat = seq_dict['feat_x'].permute((1, 0, 2))            
                elif self.abs == 'enc_feat':
                    batch_feat = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
                elif self.abs == 'sim_feat':
                    enc_opt = memory[-self.feat_len_s:].transpose(0, 1)
                    dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
                    att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
                    opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)) #.permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
                    # print("opt shape", opt.shape)
                    batch_feat = nn.functional.normalize(torch.squeeze(opt, -1), p=1.0)
                elif self.abs == 'hs':
                    batch_feat = torch.squeeze(hs)
                # print("batch feat shape", batch_feat.shape)
                # print("hs shape", hs.shape)           
            
            if batch_feat.dim() <2:
                batch_feat = torch.unsqueeze(batch_feat, 0)

            #batch_feat = torch.flatten(batch_feat, start_dim=1) ##os??
            if batch_feat.shape[1] == self.feat_len_s:
                batch_feat = batch_feat.permute(0,2,1)

            if self.cls_type == 'cls':
                tmp_obj = self.obj_cls(batch_feat)
                out_dict.update({'pred_logits': self.cls_head(hs)[-1], 'pred_obj': tmp_obj })
            else:
                tmp_obj, h, g = self.odin_cls(batch_feat)
                # print(tmp_obj.shape) #torch.Size([16, 366])
                #torch.Size([1, 16, 1, 366]) TMP hs
                
                #if self.cls_head == None:

                out_dict.update({'pred_logits': self.cls_head(hs)[-1], 'pred_obj': tmp_obj, 'pred_h':h, 'pred_g':g})
            # forward the box prediction head
        if run_box_head:
            # forward the box prediction head
            out_dict_box, outputs_coord, flagFeat = self.forward_box_head(hs, memory, annot = annot)
            # merge results
            out_dict.update(out_dict_box)
            return out_dict, outputs_coord, batch_feat
        else:
            return out_dict, None, None, 

    def adjust(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]

