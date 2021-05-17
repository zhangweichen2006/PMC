import torch
import numpy as np
from mmdet.ops import nms
from mmdet.core import bbox2result
import mmcv

import argparse

from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors

import os.path as osp
import scipy.io as sio

import ast

def get_fusion(pkl_path, classes=2, stages=[0,1,2,3]):
    # data = mmcv.load('/home/rusu5516/project/feats/abc.pkl')
    data2 = mmcv.load(pkl_path)
    all_dets = []
    NUM_CLASS = classes
    for f_n, fr in enumerate(data2):

        bboxes, labels = [], []
        for n in range(NUM_CLASS):
            dets_list = []
            for a in stages:
                if a in [2]:
                    dets_list.append(fr[a][n][fr[a][n][:,4]<1])
                else:
                    dets_list.append(fr[a][n][fr[a][n][:,4]>0.01])
                # print(a, data2[f_n])
                # dets_list.append(data2[f_n][a][n])
            # print(dets_list)
            dets = []
            for det in dets_list:
                if det.shape[0] > 0:
                    dets.append(det)
            if len(dets) > 0:
                cls_dets = np.concatenate(dets, 0)
            else:
                continue
            cls_dets = torch.from_numpy(cls_dets).cuda()
            nms_keep = nms(cls_dets, 0.4)
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
            labels = cls_labels.new_zeros((0, NUM_CLASS+1), dtype=torch.long)

        result = bbox2result(bboxes, labels, 3)
        all_dets.append(result)

    return data2, all_dets

def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    root = '/home/wzha8158/detections'
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        # info = data['img_meta']['filename'].split('/')
        # img_name = info[-1].strip('.jpg')
        # vid = info[-2]
        # cls = info[-3]
        # if not osp.isdir(osp.join(root,cls)):
        #     os.mkdir(osp.join(root,cls))
        # if not osp.isdir(osp.join(root,cls,vid)):
        #     os.mkdir(osp.join(root,cls,vid))
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result,
                                     data_loader.dataset.img_norm_cfg)

        # output_dir =osp.join(root,cls,vid,img_name+'.mat')

        # print(output_dir)

        # sio.savemat(output_dir, mdict={'loc':boxes, 'scores': scores})

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('pkl_path')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument('--classes', default=2, type=int, help='stages')
    parser.add_argument('--stages', help='stages')
    parser.add_argument('--smallset', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    # if args.gpus == 1:
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    model = MMDataParallel(model, device_ids=[0])

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)

    stage_list = ast.literal_eval(args.stages)
    ori, outputs = get_fusion(args.pkl_path, classes=args.classes, stages=stage_list)#single_test(model, data_loader, args.show)
    # else:
    #     model_args = cfg.model.copy()
    #     model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
    #     model_type = getattr(detectors, model_args.pop('type'))
    #     outputs = parallel_test(
    #         model_type,
    #         model_args,
    #         args.checkpoint,
    #         dataset,
    #         _data_func,
    #         range(args.gpus),
    #         workers_per_gpu=args.proc_per_gpu)
    # print("ori shape",np.array(ori).shape)
    # print("new shape",np.expand_dims(np.array(outputs), axis=1))
    # print("outputs", outputs)
    dataset.eval(np.expand_dims(np.array(outputs), axis=1).tolist())

if __name__ == '__main__':
    main()

