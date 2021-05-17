import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import sample_bboxes, bbox2roi, bbox2result, multi_apply
import torch.nn.functional as F


class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        # print(neck, self.with_neck)

        if neck is not None:
            self.neck = neck
            self.neck_rgb = builder.build_neck(neck)
            self.neck_flow = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_rpn_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_bbox_head(bbox_head)
            self.bbox_head_flow = builder.build_bbox_head(bbox_head)

        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_mask_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.message_blocks = nn.ModuleList([message_block(in_channel) for in_channel in [192, 480, 832, 1024]])
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        # self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck_rgb, nn.Sequential):
                for m in self.neck_rgb:
                    m.init_weights()
                for m in self.neck_flow:
                    m.init_weights()
            else:
                self.neck_rgb.init_weights()
                self.neck_flow.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            feat_mix = []
            for r, f, m in zip(x[0], x[1], self.message_blocks):
                feat_mix.append(r + m(f))
            feat_rgb = self.neck_rgb(feat_mix)
            feat_flow = self.neck_flow(x[1])
        return feat_rgb, feat_flow

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      gt_masks=None,
                      proposals=None):
        losses = dict()

        # print('start')

        num, _, height, width = img.size()

        x, y = self.extract_feat(img.view(num, -1, 3, height, width)[:,1:,:,:,:])

        # print('extract done')

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_proposals(*proposal_inputs)
        else:
            proposal_list = proposals

        # print('rpn done')

        if self.with_bbox:
            (pos_proposals, neg_proposals, pos_assigned_gt_inds, pos_gt_bboxes,
             pos_gt_labels) = multi_apply(
                 sample_bboxes,
                 proposal_list,
                 gt_bboxes,
                 gt_bboxes_ignore,
                 gt_labels,
                 cfg=self.train_cfg.rcnn)
            (labels, label_weights, bbox_targets,
             bbox_weights) = self.bbox_head.get_bbox_target(
                 pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels,
                 self.train_cfg.rcnn)

            rois = bbox2roi([
                torch.cat([pos, neg], dim=0)
                for pos, neg in zip(pos_proposals, neg_proposals)
            ])
            # TODO: a more flexible way to configurate feat maps
            roi_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)

            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, labels,
                                            label_weights, bbox_targets,
                                            bbox_weights)
            losses.update(loss_bbox)

            roi_feats_flow = self.bbox_roi_extractor(
                y[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score_flow, bbox_pred_flow = self.bbox_head_flow(roi_feats_flow)

            loss_bbox_flow = self.bbox_head_flow.loss(cls_score_flow, bbox_pred_flow, labels,
                                            label_weights, bbox_targets,
                                            bbox_weights, True)
            losses.update(loss_bbox_flow)

        if self.with_mask:
            mask_targets = self.mask_head.get_mask_target(
                pos_proposals, pos_assigned_gt_inds, gt_masks,
                self.train_cfg.rcnn)
            pos_rois = bbox2roi(pos_proposals)
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            torch.cat(pos_gt_labels))
            losses.update(loss_mask)

        # print('done')

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        num, _, height, width = img.size()

        x,y = self.extract_feat(img.view(num, -1, 3, height, width)[:,1:,:,:,:])

        # x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta,
            self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results


class message_block(nn.Module):
    def __init__(self, in_channels):
        super(message_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, int(in_channels/4), kernel_size=1)
        self.conv2 = nn.Conv2d(int(in_channels/4), in_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x