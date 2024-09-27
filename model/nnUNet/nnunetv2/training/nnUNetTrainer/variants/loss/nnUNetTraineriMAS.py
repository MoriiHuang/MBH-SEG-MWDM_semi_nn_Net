import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch.nn import functional as F


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # #  1. get training criterion
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def get_criterion(cfg):
    cfg_criterion = cfg["criterion"]
    aux_weight = (
        cfg["net"]["aux_loss"]["loss_weight"]
        if cfg["net"].get("aux_loss", False)
        else 0
    )
    ignore_index = cfg["dataset"]["ignore_label"]
    if cfg_criterion["type"] == "ohem":
        criterion = CriterionOhem(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )
    else:
        criterion = Criterion(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )

    return criterion


class Criterion(nn.Module):
    def __init__(self, aux_weight, ignore_index=255, use_weight=False):
        super(Criterion, self).__init__()
        self._aux_weight = aux_weight
        self._ignore_index = ignore_index
        self.use_weight = use_weight
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            weights = torch.FloatTensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ).cuda()
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self._criterion1 = nn.CrossEntropyLoss(
                ignore_index=ignore_index, weight=weights
            )

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )
            if self.use_weight:
                loss1 = self._criterion(main_pred, target) + self._criterion1(
                    main_pred, target
                )
            else:
                loss1 = self._criterion(main_pred, target)
            loss2 = self._criterion(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion(preds, target)
        return loss


class CriterionOhem(nn.Module):
    def __init__(
        self,
        aux_weight,
        thresh=0.7,
        min_kept=100000,
        ignore_index=255,
        use_weight=False,
    ):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh, min_kept, use_weight
        )
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (
            factor * factor
        )  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = (
            torch.from_numpy(input_label.reshape(target.size()))
            .long()
            .cuda(target.get_device())
        )

        return new_target

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=256, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # #  2. calculate MIOU for each instance
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.lst_num_per_cls = np.zeros((num_classes,))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def _get_num_per_class(self, label_true):
        res = []
        for i in range(0, self.num_classes):
            res.append((label_true==i).sum())
        return np.array(res)
        
    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
            self.lst_num_per_cls += self._get_num_per_class(lt.flatten())

    def evaluate(self, flag_cls_weight=False, flag_ignore_background=False):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        if flag_cls_weight:
            if flag_ignore_background:
                tmp_weight = self.lst_num_per_cls
                tmp_weight[0] = 0
                tmp_weight = tmp_weight / (tmp_weight.sum())
            else:
                tmp_weight = self.lst_num_per_cls / (self.lst_num_per_cls.sum())
            return iu, np.nansum(iu * tmp_weight)
        else:
            return iu, np.nanmean(iu)

def obtain_meanIOU_for_each_instance_in_batch(m_pred, m_gt, num_cls, flag_weight=False, flag_ignore_bg=False):
    assert m_pred.shape == m_gt.shape
    res_miou = []
    if len(m_pred.shape) == 3:
        for each_i in range(len(m_pred)):
            tmp_metric = meanIOU(num_cls)
            tmp_metric.add_batch(m_pred[each_i], m_gt[each_i])
            res_miou.append(tmp_metric.evaluate(flag_cls_weight=flag_weight, flag_ignore_background=flag_ignore_bg )[-1])
    else:
        raise ValueError
    # return np.array(res_miou)
    return np.array(res_miou, dtype=np.float32)


def obtain_meanIOU_for_each_instance_in_batch_old(m_pred, m_gt, num_cls):
    assert m_pred.shape == m_gt.shape
    res_miou = []
    if len(m_pred.shape) == 3:
        for each_i in range(len(m_pred)):
            tmp_metric = meanIOU(num_cls)
            tmp_metric.add_batch(m_pred[each_i], m_gt[each_i])
            res_miou.append(tmp_metric.evaluate()[-1])
            del tmp_metric
    else:
        raise ValueError
    # return np.array(res_miou)
    return np.array(res_miou, dtype=np.float32)


def compute_ulb_hardness_miou(current_pred,  teacher_pred, num_class,
            flag_using_cls_weighted_iou=True,
            flag_ignoring_background=True):
            
    hardness_tea = obtain_meanIOU_for_each_instance_in_batch(current_pred.cpu().numpy(), teacher_pred.cpu().numpy(), 
                num_class, flag_weight=flag_using_cls_weighted_iou, 
                flag_ignore_bg=flag_ignoring_background)
    hardness_stu = obtain_meanIOU_for_each_instance_in_batch(teacher_pred.cpu().numpy(), current_pred.cpu().numpy(), 
                num_class, flag_weight=flag_using_cls_weighted_iou, 
                flag_ignore_bg=flag_ignoring_background)
    # if flag_ignoring_background:
    #     hardness_tea[hardness_tea<0.002] = 1.0
    #     hardness_stu[hardness_stu<0.002] = 1.0

    return hardness_tea, hardness_stu

def compute_ulb_hardness_all(stu_outputs, tea_outputs, thresh=0.95,
            flag_using_cls_weighted_iou=True,
            flag_ignoring_background=True):
    batch_size, num_class, h, w = stu_outputs.shape
    with torch.no_grad():
        prob_stu = torch.softmax(stu_outputs.detach(), dim=1)
        prob_tea = torch.softmax(tea_outputs.detach(), dim=1)

        max_prob_stu, pred_labl_stu = torch.max(prob_stu, dim=1)
        max_prob_tea, pred_labl_tea = torch.max(prob_tea, dim=1)

        # hardness 1: iou
        hardness_iou_tea, hardness_iou_stu = compute_ulb_hardness_miou(pred_labl_stu, pred_labl_tea, num_class, 
            flag_using_cls_weighted_iou=flag_using_cls_weighted_iou, 
            flag_ignoring_background=flag_ignoring_background)

        hardness_iou_tea = torch.from_numpy(hardness_iou_tea).cuda()
        hardness_iou_stu = torch.from_numpy(hardness_iou_stu).cuda()
        
        # hardness 2: high-ratio
        hardness_ratio_tea = max_prob_tea.ge(thresh).float().mean(dim=[1,2])
        hardness_ratio_stu = max_prob_stu.ge(thresh).float().mean(dim=[1,2])
# =================================================================================================================================================================
        # orginal
        hardness_v1 = (hardness_iou_tea * hardness_ratio_tea + hardness_iou_stu * hardness_ratio_stu)/ 2
        # test new
        # hardness_v1 = (hardness_ratio_tea + hardness_ratio_stu)/ 2
# =================================================================================================================================================================
        tmp_ratio = hardness_ratio_tea + hardness_ratio_stu
        hardness_v2 =  (hardness_ratio_tea / tmp_ratio) * hardness_iou_tea + (hardness_ratio_stu / tmp_ratio) * hardness_iou_stu
    return hardness_v1, hardness_v2, hardness_ratio_stu

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # #  3. calculate unsupervised loss
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def compute_unsupervised_loss_by_threshold(predict, target, logits, thresh=0.95, hardness_tensor=None):
    batch_size, num_class, h, w = predict.shape
    thresh_mask = logits.ge(thresh).bool() * (target != 255).bool()
    target[~thresh_mask] = 255
    loss = F.cross_entropy(predict, target, ignore_index=255, reduction="none")
    if hardness_tensor is None:
        return loss.mean()
    loss = loss.mean(dim=[1,2])
    assert loss.shape == hardness_tensor.shape, "wrong hardness calculation!"
    loss *= hardness_tensor
    return loss.mean()

import numpy as np
import random
import torch
import scipy.stats as stats


def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    
    
#     print(cx,"\n", cy, "\n", cx - cut_w // 2)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
#     print("="*20)
#     print(cut_w, cut_h)
#     print([(x,y) for x,y in zip(bbx2-bbx1, bby2-bby1)])

    return bbx1, bby1, bbx2, bby2


def rand_bbox_by_v(size, v=0.2):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = v
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/16), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/16), high=H)
    
#     print(cx,"\n", cy, "\n", cx - cut_w // 2)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
#     print("="*20)
#     print(cut_w, cut_h)
#     print([(x,y) for x,y in zip(bbx2-bbx1, bby2-bby1)])

    return bbx1, bby1, bbx2, bby2


# # # # # # # # # # # # # # # # # # # # # 
# # 1.1 cutmix using beta distr
# # # # # # # # # # # # # # # # # # # # # 
def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    
    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    # print(u_rand_index)
    
    # get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        # label is of 3 dimensions
#         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_image, unlabeled_mask, unlabeled_logits

    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # 1.2 cutmix using hardness (cut easy for hard, cut hard for easy)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def cut_mix_by_hardness_beta(unlabeled_image, unlabeled_mask, unlabeled_logits,
                           hardness=0.99):
    # note
    # in this implementation, the larger hardness means the instance is easier to learn
    # # # # # # # # # # # # # # # # # # 
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    
    # 1. get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # 2. get hardness
    if (isinstance(hardness, list) or isinstance(hardness, tuple)) and len(hardness) == unlabeled_image.shape[0]:
        hardness_lst = np.array(hardness)
        flag_different_hardness = True
    elif isinstance(hardness, float):
        flag_different_hardness = False
    else:
        flag_different_hardness = False
        raise ValueError
    
    # 3. get index
    if flag_different_hardness:
        tmp_dict = {x:y for x,y in zip(np.argsort(hardness_lst), np.argsort(hardness_lst)[::-1])}
        u_rand_index = [tmp_dict[i] for i in range(len(hardness_lst))]
    else:
        u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 4. apply  cutmix
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    
    # return mixed results
    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits



# # # # # # # # # # # # # # # # # # # # # 
# # 2.1 cutmix using uniform range
# # # # # # # # # # # # # # # # # # # # # 
def cut_mix_using_v(unlabeled_image, unlabeled_mask, unlabeled_logits, scale=[0.3, 0.7]):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    
    # 0. get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    # print(u_rand_index)
    
    # 1. get box
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_by_v(unlabeled_image.size(), v=v)
    
    # 2. cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_image, unlabeled_mask, unlabeled_logits

    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # 2.2 cutmix using hardness for adjusting trigger probability
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def cut_mix_by_hardness_for_prob(unlabeled_image, unlabeled_mask, unlabeled_logits, 
                                 hardness=0.99, max_prob=0.8, scale=[0.3, 0.7], flag_hardness_batch=True):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    
    # 0. get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 1. get box
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_by_v(unlabeled_image.size(), v=v)
    
    # 2. get hardness
    if (isinstance(hardness, list) or isinstance(hardness, tuple)) and len(hardness) == unlabeled_image.shape[0]:
        hardness_lst = np.array(hardness)
    elif isinstance(hardness, float):
        hardness_lst = np.array([hardness] * unlabeled_image.shape[0])
    else:
        raise ValueError
    hardness_lst *= max_prob
    
    # 3. judge batch-hardness
    if flag_hardness_batch:
        avg_hardness = np.mean(hardness_lst)
        if np.random.uniform() > avg_hardness:
            # print("Not apply cutmix - batch")
            return unlabeled_image, unlabeled_mask, unlabeled_logits
        else:
            pass
            # print("will apply cutmix - batch")
    
    # 4. apply  cutmix
    for i in range(0, mix_unlabeled_image.shape[0]):
        # judge instance-hardness
        if not flag_hardness_batch and np.random.uniform() > hardness_lst[i]:
            continue
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_image, unlabeled_mask, unlabeled_logits

    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # 2.3. cutmix using hardness (cut easy for hard, cut hard for easy)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def get_adpative_magnituide(v_min, v_max, confidence, flag_tough_max=True, sigma=None):
    assert 0 <= confidence <= 1
    # mean
    if flag_tough_max:
        var_mu = v_min + (v_max - v_min) * confidence 
    else:
        var_mu = v_max - (v_max - v_min) * confidence 
    # sigma
    if sigma is None:
        var_sigma = max(var_mu - v_min, v_max - var_mu)
        var_sigma /=3.0
    else:
        var_sigma = sigma
    # print("="*10,f"mu:{var_mu}, std:{var_sigma}")
    # truncated norm
    a = (v_min - var_mu) / var_sigma
    b = (v_max - var_mu) / var_sigma
    rv = stats.truncnorm(a,b, loc=var_mu, scale=var_sigma)
    return rv.rvs()


def cut_mix_by_hardness(unlabeled_image, unlabeled_mask, unlabeled_logits,
                           hardness=0.99, scale=[0.55, 0.95], 
                           flag_hardness_random=False, 
                           flag_hardness_gaussion=False):
    # note
    # in this implementation, the larger hardness means the instance is easier to learn
    # # # # # # # # # # # # # # # # # # 
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    
    # 1. get hardness
    avg_hardness = None
    if (isinstance(hardness, list) or isinstance(hardness, tuple)) and len(hardness) == unlabeled_image.shape[0]:
        hardness_lst = np.array(hardness)
        flag_different_hardness = True
        avg_hardness = hardness_lst.mean()
    elif isinstance(hardness, float):
        flag_different_hardness = False
        avg_hardness = hardness
    else:
        flag_different_hardness = False
        raise ValueError
    
    # 2. get box
    min_v, max_v = min(scale), max(scale)
    if flag_hardness_gaussion:
        v = get_adpative_magnituide(min_v, max_v, avg_hardness, flag_tough_max=True)
    else:
        v = float(max_v - min_v) * random.random()
        if flag_hardness_random:
            v *= avg_hardness
        v += min_v

    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_by_v(unlabeled_image.size(), v=v)
    
    
    
    # 3. get index
    if flag_different_hardness:
        tmp_dict = {x:y for x,y in zip(np.argsort(hardness_lst), np.argsort(hardness_lst)[::-1])}
        u_rand_index = [tmp_dict[i] for i in range(len(hardness_lst))]
    else:
        u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 4. apply  cutmix
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    
    # return mixed results
    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits