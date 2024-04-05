from . import EXOTActor
import torch


class EXOTSTActor(EXOTActor):
    """ Actor for training the STARK-ST(Stage2)"""
    def __init__(self, net, objective, loss_weight, settings, loss_type):
        super().__init__(net, objective, loss_weight, settings, loss_type)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.exit_flag = loss_type

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict, flagFeat = self.forward_pass(data, run_box_head=False, run_cls_head=True)

        # process the groundtruth label
        labels = data['label'].view(-1)  # (batch, ) 0 or 1
        objlabels = data['test_class'].view(-1)
        loss, status = self.compute_losses(out_dict, labels, objlabels)
        # self.cal_ood(data, out_dict)
        return loss, status

    def compute_losses(self, pred_dict, labels, objlabels, return_status=True):
        clsloss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)
        objloss = self.loss_weight['objcls']*torch.mean(labels*self.objective['objcls'](torch.squeeze(pred_dict["pred_obj"]), objlabels))
        objidx = torch.argmax(torch.squeeze(pred_dict["pred_obj"]).detach(), dim=1)
        objaccuracy = torch.count_nonzero(labels*(objlabels == objidx))/torch.count_nonzero(labels)

        loss = clsloss+objloss
        if return_status:
            # status for log
            status = {
                "cls_loss": clsloss.item(),
                "obj_loss": objloss.item(),
                "total_loss": loss.item(),
                'obj_accuracy': objaccuracy.item()}
            return loss, status
        else:
            return loss
