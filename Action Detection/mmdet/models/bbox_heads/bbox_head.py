import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (delta2bbox, multiclass_nms, bbox_target,
                        weighted_cross_entropy, weighted_smoothl1, accuracy)


class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_bbox_target(self, pos_proposals, neg_proposals, pos_gt_bboxes,
                        pos_gt_labels, rcnn_train_cfg):
        reg_num_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_num_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def loss(self, cls_score, bbox_pred, labels, label_weights, bbox_targets,
             bbox_weights, loss_str=None):
        losses = dict()
        if loss_str is not None:
            # print("loss_str", loss_str)
            # print("labels",labels)
            # print("label_weights",label_weights)
            # print("bbox_targets",bbox_targets)
            # print("bbox_weights",bbox_weights)
            if cls_score is not None:
                losses['loss_cls_{}'.format(loss_str)] = weighted_cross_entropy(
                    cls_score, labels, label_weights)
                losses['acc_{}'.format(loss_str)] = accuracy(cls_score, labels)
            if bbox_pred is not None:
                losses['loss_reg_{}'.format(loss_str)] = weighted_smoothl1(
                    bbox_pred,
                    bbox_targets,
                    bbox_weights,
                    avg_factor=bbox_targets.size(0))
        else:
            if cls_score is not None:
                losses['loss_cls'] = weighted_cross_entropy(
                    cls_score, labels, label_weights)
                losses['acc'] = accuracy(cls_score, labels)
            if bbox_pred is not None:
                losses['loss_reg'] = weighted_smoothl1(
                    bbox_pred,
                    bbox_targets,
                    bbox_weights,
                    avg_factor=bbox_targets.size(0))
        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       num=1,
                       train_box=False,
                       rescale=False,
                       nms_cfg=None, list_out=False, score_thr=0, nms_th=0):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:]
            # TODO: add clip here

        if rescale:
            bboxes /= scale_factor

        bboxes_list = []
        scores_list = []
        for i in range(num):
            pick = rois[:,0] == i
            bboxes_list.append(bboxes[pick,:])
            scores_list.append(scores[pick,:])

        

        if nms_cfg is None:
            if list_out:
                return bboxes_list, scores_list, bboxes, scores
            return bboxes, scores
        else:

            if score_thr == 0:
                score_thr = nms_cfg.score_thr

            if nms_th == 0:
                nms_th = nms_cfg.nms_thr
            if list_out:
                det_bboxes_list = []
                det_labels_list = []
                for bboxes, scores in zip(bboxes_list, scores_list):
                    det_bboxes, det_labels = multiclass_nms(
                        bboxes, scores, score_thr, nms_th, train_box,
                        nms_cfg.max_per_img)
                    det_bboxes_list.append(det_bboxes)
                    det_labels_list.append(det_labels)

                return det_bboxes_list, det_labels_list, bboxes_list, scores_list
            else:
                det_bboxes, det_labels = multiclass_nms(
                    bboxes, scores, score_thr, nms_th, 
                    max_num=nms_cfg.max_per_img)

                return det_bboxes, det_labels, bboxes, scores
