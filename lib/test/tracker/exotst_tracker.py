from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
from copy import deepcopy
# for debug
import cv2
import os
from lib.utils.merge import merge_template_search, merge_template_search_odin
from lib.models.exot import build_exotst_odin, build_exotst_cls
from lib.test.tracker.stark_utils import PreprocessorX, Preprocessor, PreprocessorOdin
from lib.utils.box_ops import clip_box
from torch.autograd import Variable
import numpy as np
from lib.utils.misc import NestedTensor
from lib.test.evaluation.environment import env_settings


class EXOTSTTracker(BaseTracker):
    def __init__(self, params, dataset_name):
        super(EXOTSTTracker, self).__init__(params)
        self.abs = params.cfg.MODEL.HEAD_ABS

        if params.cfg.MODEL.ODIN_TYPE == 'cls':
            network = build_exotst_cls(params.cfg)
        else:
            network = build_exotst_odin(params.cfg)

        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False) #True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.device = 'cuda'
        self.network.eval()
        self.preprocessor_odin = PreprocessorOdin()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # template update
        self.z_dict1 = {}
        self.z_dict_list = []
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)
        self.num_extra_template = len(self.update_intervals)

    def initialize(self, image, info: dict):
        # initialize z_dict_list
        self.z_dict_list = []
        # get the 1st template
        z_patch_arr1, _, z_amask_arr1 = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                      output_sz=self.params.template_size)
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1)
        # get the complete z_dict_list
        self.z_dict_list.append(self.z_dict1)
        for i in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # get the t-th search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = self.z_dict_list + [x_dict]
            seq_dict = merge_template_search(feat_dict_list, return_search=True, return_template=True)
            # run the transformer
            out_dict, _, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        # update template
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > 0.5:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                self.z_dict_list[idx+1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame
        if info!=None:
            objlabels = info['object_class']
            objname = info['object_name']
            vislabels = info['target_visible']
        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)

        if self.save_all_boxes and info != None:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score,
                    "objgt": objlabels,
                    "visgt": vislabels}
        elif self.save_all_boxes and info == None:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score}
        elif info !=None:
            return {"target_bbox": self.state,
                    "conf_score": conf_score,
                    "objgt": objlabels,
                    "visgt": vislabels}
        else:
            return {"target_bbox": self.state,
                    "conf_score": conf_score}

    def odin_track(self, image, epsilon, version, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # get the t-th search region        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search_grad, search_mask = self.preprocessor_odin.process(x_patch_arr, x_amask_arr)
        '''
        [W accumulate_grad.h:185] Warning: grad and param do not obey the gradient layout contract. This is not an error, but may impair performance.
        grad.sizes() = [256, 1024, 1, 1], strides() = [1024, 1, 1, 1]
        param.sizes() = [256, 1024, 1, 1], strides() = [1024, 1, 1024, 1024] (function operator())
        '''
        search = NestedTensor(search_grad, search_mask)

        #with torch.no_grad():
        x_dict = self.network.forward_backbone(search)
        # merge the template and the search
        feat_dict_list = self.z_dict_list + [x_dict]
        seq_dict = merge_template_search(feat_dict_list, return_search=True, return_template=True)

        # run the transformer
        out_dict, batch_feat_ori, hs, enc_mem = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]       

        ###################################
        ### ODIN ###
        ###################################
        # ODIN 
        grads = {}
        def save_grad(name):
            #print("Call save grad!!")
            def hook(grad):
                #print("inside hook", grad)
                grads[name] = grad.detach()
            return hook
        # print(search_grad.shape) (1,3,320,320)
        images_grad = batch_feat_ori  #seq_dict['feat_x'].permute((1, 0, 2))
        images_grad.register_hook(save_grad('batch_feat'))   

        if self.cfg.MODEL.ODIN_TYPE=='cls':
            odin_logits = out_dict['pred_obj'].requires_grad_()
        else:
            if version == 'ori':
                odin_logits = out_dict['pred_obj'].requires_grad_()
            elif version == 'h':
                odin_logits = out_dict['pred_h'].requires_grad_()
            elif version == 'g':
                odin_logits = out_dict['pred_g'].requires_grad_()
        # print(odin_logits.shape, "ODIN") #torch.Size([400, 1, 366]) ODIN
        # images_grad = search_grad.requires_grad_()   
        # print(images_grad) 
        # if len(odin_logits.shape) >2:
        # print(odin_logits.shape, "FIRST") # 1, 1, 1, 366
        odin_logits = torch.squeeze(odin_logits, dim=1)
        # print(odin_logits.shape, "LOGITS") # 1, 366
        if version == 'ori' or version=='h':
            odin_loss = torch.amax(odin_logits, dim=1)
        else:
            odin_loss = odin_logits
        # print("ODIN loss shpae", odin_loss.shape) 1, 1
        # odin_loss = - odin_loss
        odin_loss = -torch.mean(odin_loss, 0, keepdim=True)
        # print(odin_loss, odin_loss.shape) # 1, 1

        odin_loss.backward()

        # Calculate the gradients of the scores with respect to the inputs.
        with torch.no_grad():
            # gradient = torch.ge(images_grad.grad, 0)
            gradient = torch.ge(grads['batch_feat'], 0)
            gradient = (gradient.float() - 0.5) *2
            gradient[::, 0] = (gradient[::, 0] )/self.cfg.DATA.STD[0]
            gradient[::, 1] = (gradient[::, 1] )/self.cfg.DATA.STD[1]
            gradient[::, 2] = (gradient[::, 2] )/self.cfg.DATA.STD[2]

        # Perturb the inputs and derive new mean score.
        # test_ds_var.assign_add(epsilon * gradients)
        # static_tensor = torch.as_tensor(images)
        static_tensor = images_grad.data - epsilon * gradient
        # static_tensor = torch.clamp(static_tensor, 0., 255.)
        # images = NestedTensor(static_tensor, search_mask)
        
        # with torch.no_grad():
        #     x_dict = self.network.forward_backbone(images)
        #     # merge the template and the search
        #     feat_dict_list = self.z_dict_list + [x_dict]
        #     seq_dict = merge_template_search(feat_dict_list, return_search=True, return_template=True)
        #     # run the transformer
        #     out_dict, _, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        with torch.no_grad():
            out_dict, _, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True, batch_feat=static_tensor)

        if self.cfg.MODEL.ODIN_TYPE=='cls':
            new_scores = out_dict['pred_obj']
        else:        
            if version == 'ori':
                new_scores = out_dict['pred_obj']#.sigmoid()
            elif version == 'h':
                new_scores = out_dict['pred_h']  # .sigmoid()
            elif version == 'g':
                new_scores = out_dict['pred_g']  # .sigmoid()
        new_scores = -torch.amax(new_scores, dim=1)
        meanscore = torch.mean(new_scores).cpu().detach().numpy() 
        #new_mean_score = self.perturb_images(search_grad, search_mask, out_dict, epsilon)


        ######################################################################################

        
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        

        for idx, update_i in enumerate(self.update_intervals):
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            
            # TODO update with ood conf score 
            if self.frame_id % update_i == 0 and conf_score > 0.5:
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                self.z_dict_list[idx+1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score,
                    "odin_meanscore": meanscore}
        else:
            return {"target_bbox": self.state,
                    "conf_score": conf_score,
                    "odin_meanscore": meanscore}

    def odin_new_track(self, image, epsilon, version, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # get the t-th search region        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search_grad, search_mask = self.preprocessor_odin.process(x_patch_arr, x_amask_arr)
        '''
        [W accumulate_grad.h:185] Warning: grad and param do not obey the gradient layout contract. This is not an error, but may impair performance.
        grad.sizes() = [256, 1024, 1, 1], strides() = [1024, 1, 1, 1]
        param.sizes() = [256, 1024, 1, 1], strides() = [1024, 1, 1024, 1024] (function operator())
        '''
        #search_grad =search_grad.requires_grad_()
        search = NestedTensor(search_grad, search_mask)
        search.require_grad = True

        # with torch.no_grad():
        x_dict = self.network.forward_backbone(search)
        # merge the template and the search
        feat_dict_list = self.z_dict_list + [x_dict]
        seq_dict = merge_template_search_odin(feat_dict_list, return_search=True, return_template=True)

        # run the transformer
        out_dict, batch_feat_ori, hs, enc_mem = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]       

        ###################################
        ### ODIN ###
        ###################################
        # ODIN 
        # print(search_grad.shape) (1,3,320,320)
        feat_len_s = 400
        grads = {}
        def save_grad(name):
            #print("Call save grad!!")
            def hook(grad):
                #print("inside hook", grad)
                grads[name] = grad.detach()
            return hook
        # if self.abs == 'backbone':
        #     # images_grad = search_grad.requires_grad_()
        #     images_grad = batch_feat_ori  #seq_dict['feat_x'].permute((1, 0, 2))
        #     images_grad.register_hook(save_grad('seq_dict_feat_x'))           
        # elif self.abs == 'enc_feat':
        #     batch_feat = enc_mem[-feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
        #     images_grad = batch_feat.requires_grad_()
        #     images_grad.register_hook(save_grad('seq_dict_feat_x'))
        # elif self.abs == 'sim_feat':
        #     enc_opt = enc_mem[-feat_len_s:].transpose(0, 1)
        #     dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
        #     att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
        #     opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)) #.permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        #     batch_feat = torch.nn.functional.normalize(torch.squeeze(opt, -1), p=1.0)
        #     images_grad = batch_feat.requires_grad_()
        # elif self.abs == 'hs':
        #     images_grad = torch.squeeze(batch_feat_ori)
            #images_grad = tmp.requires_grad_()
            # images_grad.register_hook(save_grad('hs'))
        
            #images_grad = batch_feat.requires_grad_()
        images_grad = batch_feat_ori  #seq_dict['feat_x'].permute((1, 0, 2))
        images_grad.register_hook(save_grad('batch_feat'))   

        if self.cfg.MODEL.ODIN_TYPE=='cls':
            odin_logits = out_dict['pred_obj'].requires_grad_()
        else:
            if version == 'ori':
                odin_logits = out_dict['pred_obj'].requires_grad_()
            elif version == 'h':
                odin_logits = out_dict['pred_h'].requires_grad_()
            elif version == 'g':
                odin_logits = out_dict['pred_g'].requires_grad_()   
        # print(odin_logits.shape, "ODIN") #torch.Size([400, 1, #cls]) ODIN
        # images_grad = search_grad.requires_grad_()   
        # print(images_grad) 
        # if len(odin_logits.shape) >2:
        # print(odin_logits.shape, "FIRST") # 1, 1, 1, #cls
        odin_logits = torch.squeeze(odin_logits, dim=1)
        # print(odin_logits.shape, "LOGITS") # 1, #cls
        if version == 'ori' or version=='h':
            odin_loss = torch.amax(odin_logits, dim=1)
        else:
            odin_loss = odin_logits
        # print("ODIN loss shpae", odin_loss.shape) 1, 1
        # odin_loss = - odin_loss
        odin_loss = -torch.mean(odin_loss, 0, keepdim=True)
        # print(odin_loss, odin_loss.shape) # 1, 1

        odin_loss.backward()

        # Calculate the gradients of the scores with respect to the inputs.
        with torch.no_grad():
            # print(grads)
            # print(grads['batch_feat'])
            # gradient = torch.ge(images_grad.grad, 0)
            gradient = torch.ge(grads['batch_feat'], 0)
            gradient = (gradient.float() - 0.5) *2
            gradient[::, 0] = (gradient[::, 0] )/self.cfg.DATA.STD[0]
            gradient[::, 1] = (gradient[::, 1] )/self.cfg.DATA.STD[1]
            gradient[::, 2] = (gradient[::, 2] )/self.cfg.DATA.STD[2]

        # Perturb the inputs and derive new mean score.
        # test_ds_var.assign_add(epsilon * gradients)
        # static_tensor = torch.as_tensor(images)
        static_tensor = images_grad.data - epsilon * gradient
        #static_tensor = torch.clamp(static_tensor, 0., 255.)

        #if self.abs =='backbone':
        #    images = NestedTensor(static_tensor, search_mask)        
        #    with torch.no_grad():
        #        x_dict = self.network.forward_backbone(images)
        #        # merge the template and the search
        #        feat_dict_list = self.z_dict_list + [x_dict]
        #        seq_dict_img = merge_template_search(feat_dict_list, return_search=True, return_template=True)
        #        batch_feat = seq_dict_img['feat_x'].permute((1,0,2))
        #        # run the transformer
        #        out_dict, _, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True, batch_feat=batch_feat)
        #elif self.abs=='enc_feat':
        with torch.no_grad():
            out_dict, _, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True, batch_feat=static_tensor)
        
        if self.cfg.MODEL.ODIN_TYPE=='cls':
            new_scores = out_dict['pred_obj']
        else:
            if version == 'ori':
                new_scores = out_dict['pred_obj']#.sigmoid()
            elif version == 'h':
                new_scores = out_dict['pred_h']  # .sigmoid()
            elif version == 'g':
                new_scores = out_dict['pred_g']  # .sigmoid()
        new_scores = -torch.amax(new_scores, dim=1)
        meanscore = torch.mean(new_scores).cpu().detach().numpy() 
        #new_mean_score = self.perturb_images(search_grad, search_mask, out_dict, epsilon)


        ######################################################################################

        
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        

        for idx, update_i in enumerate(self.update_intervals):
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            
            if self.frame_id % update_i == 0 and conf_score > 0.5:
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                self.z_dict_list[idx+1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score,
                    "odin_meanscore": meanscore}
        else:
            return {"target_bbox": self.state,
                    "conf_score": conf_score,
                    "odin_meanscore": meanscore}

    def odin_test_track(self, image, epsilon, version, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # get the t-th search region        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search_grad, search_mask = self.preprocessor_odin.process(x_patch_arr, x_amask_arr)
        '''
        [W accumulate_grad.h:185] Warning: grad and param do not obey the gradient layout contract. This is not an error, but may impair performance.
        grad.sizes() = [256, 1024, 1, 1], strides() = [1024, 1, 1, 1]
        param.sizes() = [256, 1024, 1, 1], strides() = [1024, 1, 1024, 1024] (function operator())
        '''
        search = NestedTensor(search_grad, search_mask)
        # search.require_grad = True

        # with torch.no_grad():
        x_dict = self.network.forward_backbone(search)
        # merge the template and the search
        feat_dict_list = self.z_dict_list + [x_dict]
        seq_dict = merge_template_search(feat_dict_list, return_search=True, return_template=True)

        # run the transformer
        out_dict, batch_feat_ori, hs, enc_mem = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]       
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        ###################################
        ### ODIN ###
        ###################################
        # ODIN 
        grads = {}
        def save_grad(name):
            #print("Call save grad!!")
            def hook(grad):
                #print("inside hook", grad)
                grads[name] = grad.detach()
            return hook
        images_grad = batch_feat_ori  #seq_dict['feat_x'].permute((1, 0, 2))
        images_grad.register_hook(save_grad('batch_feat'))   
        # print(search_grad.shape) (1,3,320,320)
        
        if self.cfg.MODEL.ODIN_TYPE=='cls':
            odin_logits = out_dict['pred_obj'].requires_grad_()
        else:
            if version == 'ori':
                odin_logits = out_dict['pred_obj'].requires_grad_()
            elif version == 'h':
                odin_logits = out_dict['pred_h'].requires_grad_()
            elif version == 'g':
                odin_logits = out_dict['pred_g'].requires_grad_()
        # print(odin_logits.shape, "ODIN") #torch.Size([400, 1, 366]) ODIN
        # images_grad = search_grad.requires_grad_()   
        # print(images_grad) 
        # if len(odin_logits.shape) >2:
        # print(odin_logits.shape, "FIRST") # 1, 1, 1, 366
        odin_logits = torch.squeeze(odin_logits, dim=1)
        # print(odin_logits.shape, "LOGITS") # 1, 366
        if version == 'ori' or version == 'h':
            odin_loss = torch.amax(odin_logits, dim=1)
        else:
            odin_loss = odin_logits
        # print("ODIN loss shpae", odin_loss.shape) 1, 1
        # odin_loss = - odin_loss
        odin_loss = -torch.mean(odin_loss, 0, keepdim=True)
        # print(odin_loss, odin_loss.shape) # 1, 1

        odin_loss.backward()

        # Calculate the gradients of the scores with respect to the inputs.
        with torch.no_grad():
            gradient = torch.ge(grads['batch_feat'], 0)
            # gradient = torch.ge(images_grad.grad, 0)
            gradient = (gradient.float() - 0.5) *2
            gradient[::, 0] = (gradient[::, 0] )/self.cfg.DATA.STD[0]
            gradient[::, 1] = (gradient[::, 1] )/self.cfg.DATA.STD[1]
            gradient[::, 2] = (gradient[::, 2] )/self.cfg.DATA.STD[2]

        # Perturb the inputs and derive new mean score.
        # test_ds_var.assign_add(epsilon * gradients)
        # static_tensor = torch.as_tensor(images)
        static_tensor = images_grad.data - epsilon * gradient
        # static_tensor = torch.clamp(static_tensor, 0., 255.)
        # images = NestedTensor(static_tensor, search_mask)
        
        # with torch.no_grad():
        #     x_dict = self.network.forward_backbone(images)
        #     # merge the template and the search
        #     feat_dict_list = self.z_dict_list + [x_dict]
        #     seq_dict = merge_template_search(feat_dict_list, return_search=True, return_template=True)
        #     # run the transformer
        #     out_dict, _, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)

        with torch.no_grad():
            out_dict, _, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True, batch_feat=static_tensor)

        if info!=None:
            objlabels = info['object_class']
            objname = info['object_name']
            vislabels = info['target_visible']
        # print(objlabels, labels, out_dict["pred_obj"].shape, out_dict['pred_logits'], out_dict['pred_logits'].shape)
        # 363 0.0 torch.Size([1, 366]) tensor([[[0.0033]]], device='cuda:0') torch.Size([1, 1, 1])
        
        if self.cfg.MODEL.ODIN_TYPE=='cls':
            objidx = torch.argmax(out_dict["pred_obj"].detach(), dim=1).item()
            objconf = torch.amax(out_dict['pred_obj'].detach(), dim=1).item()
        else:
            if version == 'ori':
                objidx = torch.argmax(out_dict["pred_obj"].detach(), dim=1).item()
                objconf = torch.amax(out_dict['pred_obj'].detach(), dim=1).item()
            elif version == 'h':
                objidx = torch.argmax(out_dict["pred_h"].detach(), dim=1).item()
                objconf = torch.amax(out_dict['pred_h'].detach(), dim=1).item()
            elif version == 'g':
                objidx = torch.argmax(out_dict["pred_g"].detach(), dim=1).item()
                objconf = torch.amax(out_dict['pred_g'].detach(), dim=1).item()

        # print("EXOT predict well?", objlabels, objidx, objidx == objlabels)

        ######################################################################################

        
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        bboxnum = self.state
        
        # predvis = conf_score>0.5
        

        for idx, update_i in enumerate(self.update_intervals):
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            
            if self.frame_id % update_i == 0 and conf_score > 0.5:
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                self.z_dict_list[idx+1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame

        # print(conf_score, objlabels, vislabels, objconf, objidx)

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)

        if self.save_all_boxes and info !=None:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": bboxnum,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score,
                    "objgt": objlabels,
                    "visgt": vislabels,
                    "objconf": objconf,
                    "predobj": objidx}

        elif self.save_all_boxes and info == None:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": bboxnum,

                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score,
                    "objconf": objconf,
                    "predobj": objidx}
        elif info !=None:
            return {"target_bbox": bboxnum,
                    "conf_score": conf_score,
                    "objgt": objlabels,
                    "visgt": vislabels,
                    "objconf": objconf,
                    "predobj": objidx}
        else:
            return {"target_bbox": bboxnum,
                    "conf_score": conf_score,
                    "objconf": objconf,
                    "predobj": objidx}
       

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return EXOTSTTracker
