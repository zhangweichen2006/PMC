from .two_stage import TwoStageDetector


class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 revep=10,
                 da_w=0.1):
        super(FasterRCNN, self).__init__(
                    backbone=backbone,
                    neck=neck,
                    rpn_head=rpn_head,
                    bbox_roi_extractor=bbox_roi_extractor,
                    bbox_head=bbox_head,
                    train_cfg=train_cfg,
                    test_cfg=test_cfg,
                    pretrained=pretrained,
                    revep=revep,
                    da_w=da_w)

    def train(self, mode=True):
        super(FasterRCNN, self).train(mode)

        
        # self.rpn_head.freeze_params()
        # self.neck_rgb.freeze_params()
        # self.neck_flow.freeze_params()
        # self.backbone.freeze_params()
        # # self.rpn_head_flow.freeze_params()
        # self.bbox_head.freeze_params()
        # self.bbox_head_flow.freeze_params()

        # for m in self.message_blocks:
        #     m.freeze_params()
