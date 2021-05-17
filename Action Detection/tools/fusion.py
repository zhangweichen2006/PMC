import torch
import numpy as np
from mmdet.ops import nms
from mmdet.core import bbox2result
import mmcv

data = mmcv.load('/home/rusu5516/project/feats/abc.pkl')
data2 = mmcv.load('/home/rusu5516/project/feats/test_abc_def.pkl')
all_dets = []
for f_n, fr in enumerate(data2):
    bboxes, labels = [], []
    for n in range(24):
        dets_list = []
        for a in [0,1,2,4]:
            if a in [2]:
                dets_list.append(fr[a][n][fr[a][n][:,4]<1])
            else:
                dets_list.append(fr[a][n][fr[a][n][:,4]>0.01])
#             dets_list.append(data2[f_n][a][n])
        dets = []
        for det in dets_list:
            if det.shape[0] > 0:
                dets.append(det)
        if len(dets) > 0:
            cls_dets = np.concatenate(dets, 0)
        else:
            continue
        cls_dets = torch.from_numpy(cls_dets).cuda()
        nms_keep = nms(cls_dets, 0.5)
        cls_dets = cls_dets[nms_keep, :]
        cls_labels = cls_dets.new_full(
            (len(nms_keep), ), n, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        # print(bboxes.shape[0])
        if bboxes.shape[0] > 100:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:100]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = cls_labels.new_zeros((0, 5))
        labels = cls_labels.new_zeros((0, ), dtype=torch.long)

    result = bbox2result(bboxes, labels, 25)
    all_dets.append(result)