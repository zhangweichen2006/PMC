import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import sample_bboxes, bbox2roi, bbox2result, multi_apply
import torch.nn.functional as F
import os.path as osp
import os
import scipy.io as sio
import pickle


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
            self.rpn_head_flow = builder.build_rpn_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_bbox_head(bbox_head)
            self.bbox_head_flow = builder.build_bbox_head(bbox_head)
            self.bbox_head_stage1 = builder.build_bbox_head(bbox_head)
            self.bbox_head_stage2 = builder.build_bbox_head(bbox_head)
            self.bbox_head_stage3 = builder.build_bbox_head(bbox_head)
            self.bbox_head_stage4 = builder.build_bbox_head(bbox_head)

        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_mask_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.message_blocks = nn.ModuleList([message_block(in_channel) for in_channel in [192, 480, 832, 1024]])
        # self.message_blocks_flow = nn.ModuleList([message_block(in_channel) for in_channel in [192, 480, 832, 1024]])
        self.message_block_m2a1 = message_block(256)
        self.message_block_m2a2 = message_block(256)
        self.message_block_a2m1 = message_block(256)
        self.message_block_m2a3 = message_block(256)
        self.message_block_a2m2 = message_block(256)
        self.message_block_a2m3 = message_block(256)
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
            self.rpn_head_flow.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
            self.bbox_head_stage1.init_weights()
            self.bbox_head_stage2.init_weights()
            self.bbox_head_stage3.init_weights()
            self.bbox_head_stage4.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            # feat_mix = []
            # feat_mix_flow = []
            # for r, f, m in zip(x[0], x[1], self.message_blocks):
            #     feat_mix.append(r + m(f))            
            # feat_rgb = self.neck_rgb(feat_mix)

            feat_rgb = self.neck_rgb(x[0])
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

            rpn_outs_flow = self.rpn_head_flow(y)
            rpn_loss_inputs_flow = rpn_outs_flow + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses_flow = self.rpn_head_flow.loss(*rpn_loss_inputs_flow, is_flow=True)
            losses.update(rpn_losses_flow)

            proposal_inputs_flow = rpn_outs_flow + (img_meta, self.test_cfg.rpn)
            proposal_list_flow = self.rpn_head_flow.get_proposals(*proposal_inputs_flow)
        else:
            proposal_list = proposals

        # print('rpn done')

        # print(gt_bboxes[0].size(), gt_labels[0].size())

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']

        if self.with_bbox:

            (pos_proposals_flow, neg_proposals_flow, pos_assigned_gt_inds_flow, pos_gt_bboxes_flow,
             pos_gt_labels_flow) = multi_apply(
                 sample_bboxes,
                 proposal_list_flow,
                 gt_bboxes,
                 gt_bboxes_ignore,
                 gt_labels,
                 cfg=self.train_cfg.rcnn)
            (labels_flow, label_weights_flow, bbox_targets_flow,
             bbox_weights_flow) = self.bbox_head_flow.get_bbox_target(
                 pos_proposals_flow, neg_proposals_flow, pos_gt_bboxes_flow, pos_gt_labels_flow,
                 self.train_cfg.rcnn)

            rois_flow = bbox2roi([
                torch.cat([pos, neg], dim=0)
                for pos, neg in zip(pos_proposals_flow, neg_proposals_flow)
            ])

            # for pos, neg in zip(pos_proposals_flow, neg_proposals_flow):
            #     print('pos:', pos.size())
            #     print('neg:', neg.size())

            # roi_feats_flow_1 = self.bbox_roi_extractor(
            #     y[:self.bbox_roi_extractor.num_inputs], rois_flow)
            # roi_feats_flow_2 = self.bbox_roi_extractor(
            #     x[:self.bbox_roi_extractor.num_inputs], rois_flow)
            # roi_feats_flow = roi_feats_flow_1 + self.message_block_a2m1(roi_feats_flow_2)
            roi_feats_flow = self.bbox_roi_extractor(
                y[:self.bbox_roi_extractor.num_inputs], rois_flow)
            cls_score_flow, bbox_pred_flow = self.bbox_head_flow(roi_feats_flow)

            det_bboxes_list_flow, det_labels_list_flow, det_rois_list_flow, det_scores_list_flow = self.bbox_head_flow.get_det_bboxes(
                rois_flow,
                cls_score_flow,
                bbox_pred_flow,
                img_shape,
                scale_factor,
                num,
                True,
                rescale=None,
                nms_cfg=self.test_cfg.rcnn, list_out=True)

            for idx, det_rois in enumerate(det_rois_list_flow):
              l = int(gt_labels[idx][0])
              det_rois_list_flow[idx] = det_rois[:,4*l:4*(l+1)]

            loss_bbox_flow = self.bbox_head_flow.loss(cls_score_flow, bbox_pred_flow, labels_flow,
                                            label_weights_flow, bbox_targets_flow,
                                            bbox_weights_flow, 'flow')
            losses.update(loss_bbox_flow)

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

            # for pos, neg in zip(pos_proposals_flow, neg_proposals_flow):
            #    print('pos:', pos.size())
            #    print('neg:', neg.size())



            rois = bbox2roi([
                torch.cat([pos, neg], dim=0)
                for pos, neg in zip(pos_proposals, neg_proposals)
            ])

            
            # TODO: a more flexible way to configurate feat maps
            # roi_feats_1 = self.bbox_roi_extractor(
            #     x[:self.bbox_roi_extractor.num_inputs], rois)
            # roi_feats_2 = self.bbox_roi_extractor(
            #     y[:self.bbox_roi_extractor.num_inputs], rois)
            # roi_feats = roi_feats_1 + self.message_block_m2a1(roi_feats_2)
            roi_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)

            det_bboxes_list, det_labels_list, det_rois_list, det_scores_list = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                num,
                True,
                rescale=None,
                nms_cfg=self.test_cfg.rcnn, list_out=True)

            for idx, det_rois in enumerate(det_rois_list):
              l = int(gt_labels[idx][0])
              det_rois_list[idx] = det_rois[:,4*l:4*(l+1)]

            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, labels,
                                            label_weights, bbox_targets,
                                            bbox_weights)
            losses.update(loss_bbox)

            #########################################################################################

            try:
                rois_stage1_list = [torch.cat([det[:,:4], det_flow[:,:4]], 0) for det, det_flow in zip(det_rois_list, det_bboxes_list_flow)]
            except:
                print(det_bboxes_list, det_bboxes_list_flow)
                raise(Exception)

            (pos_proposals_stage1, neg_proposals_stage1, pos_assigned_gt_inds_stage1, pos_gt_bboxes_stage1,
             pos_gt_labels_stage1) = multi_apply(
                 sample_bboxes,
                 rois_stage1_list,
                 gt_bboxes,
                 gt_bboxes_ignore,
                 gt_labels,
                 cfg=self.train_cfg.rcnn)
            (labels_stage1, label_weights_stage1, bbox_targets_stage1,
             bbox_weights_stage1) = self.bbox_head_stage1.get_bbox_target(
                 pos_proposals_stage1, neg_proposals_stage1, pos_gt_bboxes_stage1, pos_gt_labels_stage1,
                 self.train_cfg.rcnn)

            rois_stage1 = bbox2roi([
                torch.cat([pos, neg], dim=0)
                for pos, neg in zip(pos_proposals_stage1, neg_proposals_stage1)
            ])
            roi_feats_stage1_rgb = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois_stage1)
            roi_feats_stage1_flow = self.bbox_roi_extractor(
                y[:self.bbox_roi_extractor.num_inputs], rois_stage1)

            roi_feats_stage1 = roi_feats_stage1_rgb + self.message_block_m2a2(roi_feats_stage1_flow)

            cls_score_stage1, bbox_pred_stage1 = self.bbox_head_stage1(roi_feats_stage1)

            det_bboxes_list_stage1, det_labels_list_stage1, det_rois_list_stage1, det_scores_list_stage1 = self.bbox_head_stage1.get_det_bboxes(
                rois_stage1,
                cls_score_stage1,
                bbox_pred_stage1,
                img_shape,
                scale_factor,
                num,
                True,
                rescale=None,
                nms_cfg=self.test_cfg.rcnn, list_out=True)

            for idx, det_rois in enumerate(det_rois_list_stage1):
              l = int(gt_labels[idx][0])
              det_rois_list_stage1[idx] = det_rois[:,4*l:4*(l+1)]

            # print('proposals:',pos_proposals_stage1[0])
            # print('gt:',gt_bboxes[0])
            # print('det:',det_bboxes_list_stage1[0])

            loss_bbox_stage1 = self.bbox_head_stage1.loss(cls_score_stage1, bbox_pred_stage1, labels_stage1,
                                            label_weights_stage1, bbox_targets_stage1,
                                            bbox_weights_stage1, 'stage1')
            losses.update(loss_bbox_stage1)




            rois_stage2_list = [torch.cat([det[:,:4], det_flow[:,:4]], 0) for det, det_flow in zip(det_bboxes_list_stage1, det_rois_list_flow)]

            (pos_proposals_stage2, neg_proposals_stage2, pos_assigned_gt_inds_stage2, pos_gt_bboxes_stage2,
             pos_gt_labels_stage2) = multi_apply(
                 sample_bboxes,
                 rois_stage2_list,
                 gt_bboxes,
                 gt_bboxes_ignore,
                 gt_labels,
                 cfg=self.train_cfg.rcnn)
            (labels_stage2, label_weights_stage2, bbox_targets_stage2,
             bbox_weights_stage2) = self.bbox_head_stage2.get_bbox_target(
                 pos_proposals_stage2, neg_proposals_stage2, pos_gt_bboxes_stage2, pos_gt_labels_stage2,
                 self.train_cfg.rcnn)

            rois_stage2 = bbox2roi([
                torch.cat([pos, neg], dim=0)
                for pos, neg in zip(pos_proposals_stage2, neg_proposals_stage2)
            ])
            roi_feats_stage2_flow = self.bbox_roi_extractor(
                y[:self.bbox_roi_extractor.num_inputs], rois_stage2)
            roi_feats_stage2_rgb = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois_stage2)
            roi_feats_stage2 = roi_feats_stage2_flow + self.message_block_a2m2(roi_feats_stage2_rgb)

            cls_score_stage2, bbox_pred_stage2 = self.bbox_head_stage2(roi_feats_stage2)

            det_bboxes_list_stage2, det_labels_list_stage2, det_rois_list_stage2, det_scores_list_stage2 = self.bbox_head_stage2.get_det_bboxes(
                rois_stage2,
                cls_score_stage2,
                bbox_pred_stage2,
                img_shape,
                scale_factor,
                num,
                True,
                rescale=None,
                nms_cfg=self.test_cfg.rcnn, list_out=True)

            for idx, det_rois in enumerate(det_rois_list_stage2):
              l = int(gt_labels[idx][0])
              det_rois_list_stage2[idx] = det_rois[:,4*l:4*(l+1)]

            loss_bbox_stage2 = self.bbox_head_stage2.loss(cls_score_stage2, bbox_pred_stage2, labels_stage2,
                                            label_weights_stage2, bbox_targets_stage2,
                                            bbox_weights_stage2, 'stage2')
            losses.update(loss_bbox_stage2)





            # rois_stage3_list = [torch.cat([det[:,:4], det_flow[:,:4]], 0) for det, det_flow in zip(det_rois_list_stage1, det_bboxes_list_stage2)]


            # (pos_proposals_stage3, neg_proposals_stage3, pos_assigned_gt_inds_stage3, pos_gt_bboxes_stage3,
            #  pos_gt_labels_stage3) = multi_apply(
            #      sample_bboxes,
            #      rois_stage3_list,
            #      gt_bboxes,
            #      gt_bboxes_ignore,
            #      gt_labels,
            #      cfg=self.train_cfg.rcnn)
            # (labels_stage3, label_weights_stage3, bbox_targets_stage3,
            #  bbox_weights_stage3) = self.bbox_head_stage3.get_bbox_target(
            #      pos_proposals_stage3, neg_proposals_stage3, pos_gt_bboxes_stage3, pos_gt_labels_stage3,
            #      self.train_cfg.rcnn)

            # rois_stage3 = bbox2roi([
            #     torch.cat([pos, neg], dim=0)
            #     for pos, neg in zip(pos_proposals_stage3, neg_proposals_stage3)
            # ])
            # roi_feats_stage3_rgb = self.bbox_roi_extractor(
            #     x[:self.bbox_roi_extractor.num_inputs], rois_stage3)
            # roi_feats_stage3_flow = self.bbox_roi_extractor(
            #     y[:self.bbox_roi_extractor.num_inputs], rois_stage3)
            # roi_feats_stage3 = roi_feats_stage3_rgb + self.message_block_m2a3(roi_feats_stage3_flow)

            # cls_score_stage3, bbox_pred_stage3 = self.bbox_head_stage3(roi_feats_stage3)

            # det_bboxes_list_stage3, det_labels_list_stage3, det_rois_list_stage3, det_scores_list_stage3 = self.bbox_head_stage3.get_det_bboxes(
            #     rois_stage3,
            #     cls_score_stage3,
            #     bbox_pred_stage3,
            #     img_shape,
            #     scale_factor,
            #     num,
            #     True,
            #     rescale=None,
            #     nms_cfg=self.test_cfg.rcnn, list_out=True)


            # loss_bbox_stage3 = self.bbox_head_stage3.loss(cls_score_stage3, bbox_pred_stage3, labels_stage3,
            #                                 label_weights_stage3, bbox_targets_stage3,
            #                                 bbox_weights_stage3, 'stage3')
            # losses.update(loss_bbox_stage3)


            # rois_stage4_list = [torch.cat([det[:,:4], det_flow[:,:4]], 0) for det, det_flow in zip(det_rois_list_stage2, det_bboxes_list_stage3)]


            # (pos_proposals_stage4, neg_proposals_stage4, pos_assigned_gt_inds_stage4, pos_gt_bboxes_stage4,
            #  pos_gt_labels_stage4) = multi_apply(
            #      sample_bboxes,
            #      rois_stage4_list,
            #      gt_bboxes,
            #      gt_bboxes_ignore,
            #      gt_labels,
            #      cfg=self.train_cfg.rcnn,)
            # (labels_stage4, label_weights_stage4, bbox_targets_stage4,
            #  bbox_weights_stage4) = self.bbox_head_stage4.get_bbox_target(
            #      pos_proposals_stage4, neg_proposals_stage4, pos_gt_bboxes_stage4, pos_gt_labels_stage4,
            #      self.train_cfg.rcnn)

            # rois_stage4 = bbox2roi([
            #     torch.cat([pos, neg], dim=0)
            #     for pos, neg in zip(pos_proposals_stage4, neg_proposals_stage4)
            # ])
            # roi_feats_stage4_flow = self.bbox_roi_extractor(
            #     y[:self.bbox_roi_extractor.num_inputs], rois_stage4)
            # roi_feats_stage4_rgb = self.bbox_roi_extractor(
            #     x[:self.bbox_roi_extractor.num_inputs], rois_stage4)
            # roi_feats_stage4 = roi_feats_stage4_flow + self.message_block_a2m3(roi_feats_stage4_rgb)

            # cls_score_stage4, bbox_pred_stage4 = self.bbox_head_stage4(roi_feats_stage4)

            # det_bboxes_list_stage4, det_labels_list_stage4, det_rois_list_stage4, det_scores_list_stage4 = self.bbox_head_stage4.get_det_bboxes(
            #     rois_stage4,
            #     cls_score_stage4,
            #     bbox_pred_stage4,
            #     img_shape,
            #     scale_factor,
            #     num,
            #     True,
            #     rescale=None,
            #     nms_cfg=self.test_cfg.rcnn, list_out=True)


            # loss_bbox_stage4 = self.bbox_head_stage4.loss(cls_score_stage4, bbox_pred_stage4, labels_stage4,
            #                                 label_weights_stage4, bbox_targets_stage4,
            #                                 bbox_weights_stage4, 'stage4')
            # losses.update(loss_bbox_stage4)


            

            

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

        # det_bboxes, det_labels, original_bboxes, original_socres = self.simple_test_bboxes(
        #     x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
            # x, img_meta, proposal_list, None, rescale=rescale)

        # with open('compare.pkl', 'wb') as fid:
        #   a = det_bboxes.cpu().numpy()
        #   b = det_labels.cpu().numpy()
        #   c = original_bboxes.cpu().numpy()
        #   d = original_socres.cpu().numpy()
        #   pickle.dump([a,b,c,d], fid)

        rois = bbox2roi(proposal_list)
        # roi_feats_1 = self.bbox_roi_extractor(
        #     x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        # roi_feats_2 = self.bbox_roi_extractor(
        #     y[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        # roi_feats = roi_feats_1 + self.message_block_m2a1(roi_feats_2)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes_list, det_labels_list, det_rois_list, det_scores_list = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            train_box=True,
            rescale=None,
            nms_cfg=self.test_cfg.rcnn,
            list_out=True)

        det_bboxes, det_labels, original_bboxes, original_socres = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            train_box=False,
            rescale=True,
            nms_cfg=self.test_cfg.rcnn,
            score_thr=0.01,
            nms_th=0.5)

        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        rpn_outs_flow = self.rpn_head_flow(y)
        proposal_inputs_flow = rpn_outs_flow + (img_meta, self.test_cfg.rpn)
        proposal_list_flow = self.rpn_head_flow.get_proposals(*proposal_inputs_flow)

        # det_bboxes, det_labels, original_bboxes, original_socres = self.simple_test_bboxes(
        #     x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=None)
            # x, img_meta, proposal_list, None, rescale=rescale)

        rois_flow = bbox2roi(proposal_list_flow)
        # roi_feats_flow_1 = self.bbox_roi_extractor(
        #     y[:len(self.bbox_roi_extractor.featmap_strides)], rois_flow)
        # roi_feats_flow_2 = self.bbox_roi_extractor(
        #     x[:len(self.bbox_roi_extractor.featmap_strides)], rois_flow)
        # roi_feats_flow = roi_feats_flow_1 + self.message_block_a2m1(roi_feats_flow_2)
        roi_feats_flow = self.bbox_roi_extractor(
            y[:len(self.bbox_roi_extractor.featmap_strides)], rois_flow)
        cls_score_flow, bbox_pred_flow = self.bbox_head_flow(roi_feats_flow)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes_list_flow, det_labels_list_flow, det_rois_list_flow, det_scores_list_flow = self.bbox_head_flow.get_det_bboxes(
            rois_flow,
            cls_score_flow,
            bbox_pred_flow,
            img_shape,
            scale_factor,
            train_box=True,
            rescale=None,
            nms_cfg=self.test_cfg.rcnn,
            list_out=True)

        det_bboxes_flow, det_labels_flow, original_bboxes_flow, original_socres_flow = self.bbox_head_flow.get_det_bboxes(
            rois_flow,
            cls_score_flow,
            bbox_pred_flow,
            img_shape,
            scale_factor,
            train_box=False,
            rescale=True,
            nms_cfg=self.test_cfg.rcnn,
            score_thr=0.01,
            nms_th=0.5)

        bbox_results_flow = bbox2result(det_bboxes_flow, det_labels_flow,
                                   self.bbox_head_flow.num_classes)

        rois_stage1 = bbox2roi([
                box_fution(det1, sco, det2)[0]
                for det1, sco, det2 in zip(det_rois_list, det_scores_list, det_bboxes_list_flow)
            ])
        rois_scores_stage1 = [
                box_fution(det1, sco, det2)[1]
                for det1, sco, det2 in zip(det_rois_list, det_scores_list, det_bboxes_list_flow)
            ][0]
        roi_feats_stage1_rgb = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois_stage1)
        roi_feats_stage1_flow = self.bbox_roi_extractor(
            y[:len(self.bbox_roi_extractor.featmap_strides)], rois_stage1)
        roi_feats_stage1 = roi_feats_stage1_rgb + self.message_block_m2a2(roi_feats_stage1_flow)
        cls_score_stage1, bbox_pred_stage1 = self.bbox_head_stage1(roi_feats_stage1)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes_list_stage1, det_labels_list_stage1, det_rois_list_stage1, det_scores_list_stage1 = self.bbox_head_stage1.get_det_bboxes(
            rois_stage1,
            cls_score_stage1,
            bbox_pred_stage1,
            img_shape,
            scale_factor,
            train_box=True,
            rescale=None,
            nms_cfg=self.test_cfg.rcnn,
            list_out=True)

        det_bboxes_stage1, det_labels_stage1, original_bboxes_stage1, original_socres_stage1 = self.bbox_head_stage1.get_det_bboxes(
            rois_stage1,
            cls_score_stage1,
            bbox_pred_stage1,
            img_shape,
            scale_factor,
            train_box=False,
            rescale=True,
            nms_cfg=self.test_cfg.rcnn,
            score_thr=0.01,
            nms_th=0.5)

        bbox_results_stage1 = bbox2result(det_bboxes_stage1, det_labels_stage1,
                                   self.bbox_head_stage1.num_classes)

        rois_stage2 = bbox2roi([
                box_fution(det1, sco, det2)[0]
                for det1, sco, det2 in zip(det_rois_list_flow, det_scores_list_flow, det_bboxes_list_stage1)
            ])
        rois_scores_stage2 = [
                box_fution(det1, sco, det2)[1]
                for det1, sco, det2 in zip(det_rois_list_flow, det_scores_list_flow, det_bboxes_list_stage1)
            ][0]

        roi_feats_stage2_flow = self.bbox_roi_extractor(
            y[:len(self.bbox_roi_extractor.featmap_strides)], rois_stage2)
        roi_feats_stage2_rgb = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois_stage2)
        roi_feats_stage2 = roi_feats_stage2_flow + self.message_block_a2m2(roi_feats_stage2_rgb)
        cls_score_stage2, bbox_pred_stage2 = self.bbox_head_stage2(roi_feats_stage2)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes_list_stage2, det_labels_list_stage2, det_rois_list_stage2, det_scores_list_stage2 = self.bbox_head_stage2.get_det_bboxes(
            rois_stage2,
            cls_score_stage2,
            bbox_pred_stage2,
            img_shape,
            scale_factor,
            train_box=True,
            rescale=None,
            nms_cfg=self.test_cfg.rcnn,
            list_out=True)

        det_bboxes_stage2, det_labels_stage2, original_bboxes_stage2, original_socres_stage2 = self.bbox_head_stage2.get_det_bboxes(
            rois_stage2,
            cls_score_stage2,
            bbox_pred_stage2,
            img_shape,
            scale_factor,
            train_box=False,
            rescale=True,
            nms_cfg=self.test_cfg.rcnn,
            score_thr=0.01,
            nms_th=0.5)

        bbox_results_stage2 = bbox2result(det_bboxes_stage2, det_labels_stage2,
                                   self.bbox_head_stage2.num_classes)

        # rois_stage3 = bbox2roi([
        #         box_fution(det1, sco, det2)[0]
        #         for det1, sco, det2 in zip(det_rois_list_stage1, det_scores_list_stage1, det_bboxes_list_stage2)
        #     ])
        # rois_scores_stage3 = [
        #         box_fution(det1, sco, det2)[1]
        #         for det1, sco, det2 in zip(det_rois_list_stage1, det_scores_list_stage1, det_bboxes_list_stage2)
        #     ][0]

        # roi_feats_stage3_rgb = self.bbox_roi_extractor(
        #     x[:len(self.bbox_roi_extractor.featmap_strides)], rois_stage3)
        # roi_feats_stage3_flow = self.bbox_roi_extractor(
        #     y[:len(self.bbox_roi_extractor.featmap_strides)], rois_stage3)
        # roi_feats_stage3 = roi_feats_stage3_rgb + self.message_block_m2a3(roi_feats_stage3_flow)
        # cls_score_stage3, bbox_pred_stage3 = self.bbox_head_stage3(roi_feats_stage3)
        # img_shape = img_meta[0]['img_shape']
        # scale_factor = img_meta[0]['scale_factor']
        # det_bboxes_list_stage3, det_labels_list_stage3, det_rois_list_stage3, det_scores_list_stage3 = self.bbox_head_stage3.get_det_bboxes(
        #     rois_stage3,
        #     cls_score_stage3,
        #     bbox_pred_stage3,
        #     img_shape,
        #     scale_factor,
        #     train_box=True,
        #     rescale=None,
        #     nms_cfg=self.test_cfg.rcnn,
        #     list_out=True)
        # det_bboxes_satge3, det_labels_stage3, original_bboxes_stage3, original_socres_stage3 = self.bbox_head_stage3.get_det_bboxes(
        #     rois_stage3,
        #     cls_score_stage3,
        #     bbox_pred_stage3,
        #     img_shape,
        #     scale_factor,
        #     train_box=False,
        #     rescale=True,
        #     nms_cfg=self.test_cfg.rcnn,
        #     score_thr=0.01,
        #     nms_th=0.5)

        # bbox_results_stage3 = bbox2result(det_bboxes_satge3, det_labels_stage3,
        #                            self.bbox_head.num_classes)


        # rois_stage4 = bbox2roi([
        #         box_fution(det1, sco, det2)[0]
        #         for det1, sco, det2 in zip(det_rois_list_stage2, det_scores_list_stage2, det_bboxes_list_stage3)
        #     ])
        # rois_scores_stage4 = [
        #         box_fution(det1, sco, det2)[1]
        #         for det1, sco, det2 in zip(det_rois_list_stage2, det_scores_list_stage2, det_bboxes_list_stage3)
        #     ][0]
        # roi_feats_stage4_flow = self.bbox_roi_extractor(
        #     y[:len(self.bbox_roi_extractor.featmap_strides)], rois_stage4)
        # roi_feats_stage4_rgb = self.bbox_roi_extractor(
        #     x[:len(self.bbox_roi_extractor.featmap_strides)], rois_stage4)
        # roi_feats_stage4 = roi_feats_stage4_flow + self.message_block_a2m3(roi_feats_stage4_rgb)
        # cls_score_stage4, bbox_pred_stage4 = self.bbox_head_stage4(roi_feats_stage4)
        # img_shape = img_meta[0]['img_shape']
        # scale_factor = img_meta[0]['scale_factor']
        # det_bboxes_list_stage4, det_labels_list_stage4, det_rois_list_stage4, det_scores_list_stage4 = self.bbox_head_stage4.get_det_bboxes(
        #     rois_stage4,
        #     cls_score_stage4,
        #     bbox_pred_stage4,
        #     img_shape,
        #     scale_factor,
        #     train_box=True,
        #     rescale=None,
        #     nms_cfg=self.test_cfg.rcnn,
        #     list_out=True)

        # det_bboxes_stage4, det_labels_stage4, original_bboxes_stage4, original_socres_stage4 = self.bbox_head_stage4.get_det_bboxes(
        #     rois_stage4,
        #     cls_score_stage4,
        #     bbox_pred_stage4,
        #     img_shape,
        #     scale_factor,
        #     train_box=False,
        #     rescale=True,
        #     nms_cfg=self.test_cfg.rcnn,
        #     score_thr=0.01,
        #     nms_th=0.5)

        # bbox_results_stage4 = bbox2result(det_bboxes_stage4, det_labels_stage4,
        #                            self.bbox_head_stage4.num_classes)

        # original_scores_list = [original_socres_flow, original_socres, original_socres_stage1, original_socres_stage2, original_socres_stage3, original_socres_stage4]
        # original_bboxes_list = [original_bboxes_flow, original_bboxes, original_bboxes_stage1, original_bboxes_stage2, original_bboxes_stage3, original_bboxes_stage4]
        # rois_list = [rois_flow, rois, rois_stage1, rois_stage2, rois_stage3, rois_stage4]
        # rois_scores_list = [rois_scores_stage1, rois_scores_stage1, rois_scores_stage1, rois_scores_stage2, rois_scores_stage3, rois_scores_stage4]
        
        # print(img_meta)
        if not self.with_mask:
        #     for n_idx, type_n in enumerate(['flow/','rgb/','stage1/','stage2/','stage3/','stage4/']):
        #         root = '/home/rusu5516/rebuttal/'+type_n
        #         info = img_meta[0]['filename'].split('/')
        #         img_name = info[-1].strip('.jpg')
        #         vid = info[-2]
        #         cls = info[-3]
        #         # if not osp.isdir(osp.join(root, cls)):
        #         #     os.mkdir(osp.join(root, cls))
        #         # if not osp.isdir(osp.join(root, cls, vid)):
        #         #     os.mkdir(osp.join(root, cls, vid))
        #         output_dir_detection =osp.join(root,'detection',cls,vid,img_name+'.mat')
        #         output_dir_proposal = osp.join(root,'proposal',cls,vid,img_name+'.mat')

        # # print(output_dir)

        #         sio.savemat(output_dir_detection, mdict={'loc':original_bboxes_list[n_idx].cpu().numpy(), 'scores': original_scores_list[n_idx].cpu().numpy()})
        #         sio.savemat(output_dir_proposal, mdict={'loc':rois_list[n_idx].cpu().numpy(), 'scores': rois_scores_list[n_idx].cpu().numpy()})
            # return det_bboxes[:300,:], det_labels[:300,:]
            return bbox_results, bbox_results_flow, bbox_results_stage1, bbox_results_stage2#, bbox_results_stage3, bbox_results_stage4
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
        # self.init_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.normal_(self.conv2.weight, 0, 0.001)
        nn.init.constant_(self.conv2.bias, 0)


def box_fution(top_boxes, top_scores, bottom_boxes):
  # print(top_boxes.size())
  num, _ = top_boxes.size()
  gaps = torch.arange(num).cuda() * 3
  # print(gaps)
  # top_selected_boxes = []
  scores_idx = torch.argmax(top_scores[:,1:], dim=1).cuda()
  # for idx, b in enumerate(top_boxes):
  #   if scores_idx[idx] > 23:
  #     print(scores_idx[idx], idx)
  #   # top_selected_boxes.append(b[4*(scores_idx[idx]+1): 4*(scores_idx[idx]+2)])
  # print(top_boxes.size())
  top_selected_boxes = top_boxes.view(-1,4)[scores_idx + gaps + 1]
  top_selected_scores = top_scores.view(-1)[scores_idx + gaps + 1]
  # print(top_selected_boxes.size())
  # top_selected_boxes = torch.stack(top_selected_boxes, dim=0)
  return torch.cat([top_selected_boxes, bottom_boxes[:,:4]], 0), torch.cat([top_selected_scores, bottom_boxes[:,4]],0)