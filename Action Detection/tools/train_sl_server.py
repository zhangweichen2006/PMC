from __future__ import division

import argparse
from mmcv import Config
from mmcv.runner import obj_from_dict

from mmdet import datasets, __version__
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector

import torch

import mmcv

from mmcv.runner import load_checkpoint, parallel_test
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import detectors

import os.path as osp
import scipy.io as sio

import numpy as np

import os, errno

import matplotlib
from matplotlib.offsetbox import AnchoredText
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import copy 
from mmdet.ops import nms
from mmdet.core import bbox2result

def get_fusion(pkl_path, classes=2, stages=[0,1]):
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
    pseudo_data_dict = {}
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        info = data['img_meta'][0].data[0][0]['filename']
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

        pseudo_data_dict[info] = {'pseudo':result}

        # if show:
        #     model.module.show_result(data, result,
        #                              data_loader.dataset.img_norm_cfg)

        # output_dir =osp.join(root,cls,vid,img_name+'.mat')

        # print(output_dir)

        # sio.savemat(output_dir, mdict={'loc':boxes, 'scores': scores})

        # batch_size = data['img'][0].size(0)
        # for _ in range(batch_size):
        # if i > 10:
        #     break
        prog_bar.update()

    return results, pseudo_data_dict

# def process_results(results, data_loader):

#     pseudo_data_dict = {}

#     for i, data in enumerate(data_loader):
#         print(data)
#         info = data['img_meta'][0].data[0][0]['filename']

#         result = results[i]

#         pseudo_data = copy.deepcopy(data)
#         pseudo_data['img_meta'][0].data[0][0].update({'pseudo': result})
#         pseudo_data_dict[info] = pseudo_data
        # if i > 10:
        #     break
    
#     return pseudo_data_dict


def process_fuse(fuse_results, pseudo_data_dict, data_loader, show=False):

    new_pseudo_data_dict = pseudo_data_dict

    for i, data in enumerate(data_loader):
        # print(data)
        info = data['img_meta'][0].data[0][0]['filename']

        fuse_result = fuse_results[i]

        new_pseudo_data_dict[info].update({'fuse_pseudo':fuse_result})

        # if i > 10:
        #     break
    
    return new_pseudo_data_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args

def draw_graph(cfg, epoch_point, point_list, fuse_map_point):
    ######### draw ###########

    fig, ax = plt.subplots()

    ax.plot(epoch_point, point_list[0], 'k', label='RGB MAP',color='r')
    ax.plot(epoch_point, point_list[1], 'k', label='Flow MAP',color='g')
    ax.plot(epoch_point, fuse_map_point, 'k', label='Fuse MAP',color='k')
    if len(point_list) > 2:
        ax.plot(epoch_point, point_list[2], 'k', label='Stage1 MAP',color='b')
        ax.plot(epoch_point, point_list[3], 'k', label='Stage2 MAP',color='c')

    ax.annotate('Last Epoch Accuracy: %0.4f' % (point_list[0][-1]), xy=(1.05, 0.1), xycoords='axes fraction', size=14)

    # Now add the legend with some customizations.
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    fig.text(0.5, 0.02, 'EPOCH', ha='center')
    fig.text(0.02, 0.5, 'ACCURACY', va='center', rotation='vertical')

    plt.savefig(os.path.join(cfg.work_dir, '0.map.png'), bbox_inches='tight')
    
    fig.clf()

    plt.clf()

def main():
    args = parse_args()


    cfg = Config.fromfile(args.config)
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    cfg.gpus = args.gpus
    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # torch.save(model.state_dict(), 'pretrain_stage1.pth')
    eps = cfg.real_total_epochs
    rgb_map_point = []
    flow_map_point = []
    fuse_map_point = []

    stage1_map_point = []
    stage2_map_point = []

    global_iter = 0

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg, revep=cfg.revep, da_w=cfg.da_w)

    for e in range(cfg.real_total_epochs):
        print()
        print('Epoch {}:'.format(e+1))

        epoch_point = [(n+1) for n in range(e+1)]

        # Training Source + Target

        print('----------------------------------------------------')
        print('---------   Training Source + Target  --------------')
        print('----------------------------------------------------')

        train_dataset = obj_from_dict(cfg.data.train, datasets)
        
        train_detector(
            model,
            train_dataset,
            cfg,
            distributed=distributed,
            validate=args.validate,
            logger=logger, epoch=e)

        # Test Target and get Pseudo
 
        print('----------------------------------------------------')
        print('---------   Test Target and get Pseudo  ------------')
        print('----------------------------------------------------')

        if 'ucf' in cfg.dataset_type_pseudo.lower():
            trim = True
        else:
            trim = False
        test_dataset = obj_from_dict(cfg.data_pseudo.train, datasets, dict(test_mode=True, pseudo_mode=False, trim=trim))

        target_loader = build_dataloader(
                            test_dataset,
                            imgs_per_gpu=1,
                            workers_per_gpu=cfg.data_pseudo.workers_per_gpu,
                            num_gpus=1,
                            dist=False,
                            shuffle=False)

        test_model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg, revep=cfg.revep, da_w=cfg.da_w)
        load_checkpoint(test_model, cfg.load_from)
        test_model = MMDataParallel(test_model, device_ids=[0])
        #
        outputs, pseudo_data_dict = single_test(test_model, target_loader)
        # print("process_results...")
        # pseudo_data_dict = process_results(outputs, target_loader)
        pseudo_data_dict['epoch'] = e
        pseudo_data_dict['total_epoch'] = cfg.real_total_epochs
        print("dump results...")
        mmcv.dump(outputs, cfg.out)
        # map_list = test_dataset.eval(outputs)

        point_list = []
        eval_list = test_dataset.eval(outputs) #[0,0,0,0]#
        stage_list = [n for n in range(len(eval_list))] # [0,1]
        # print("len(test_dataset.classes)", len(test_dataset.classes))

        print('-------------------')
        print('------- FUSE ------')
        #
        pkl_path = os.path.join(cfg.work_dir, 'results.pkl')
        ori, fuse_outputs = get_fusion(pkl_path, classes=len(test_dataset.classes), stages=stage_list)

        rearranged_fuse_outputs = np.expand_dims(np.array(fuse_outputs), axis=1).tolist()
        fuse_eval_list = test_dataset.eval(rearranged_fuse_outputs)

        pseudo_data_dict = process_fuse(rearranged_fuse_outputs, pseudo_data_dict, target_loader)
        print('draw graph...')

        # print("ori, fuse", ori, fuse_outputs)
        rgb_map_point.append(float("%.4f" % eval_list[0]))
        flow_map_point.append(float("%.4f" % eval_list[1]))
        fuse_map_point.append(float("%.4f" % fuse_eval_list[0]))
        point_list.append(rgb_map_point)
        point_list.append(flow_map_point)
        if len(eval_list) > 2:
            stage1_map_point.append(float("%.4f" % eval_list[2]))
            stage2_map_point.append(float("%.4f" % eval_list[3]))
            point_list.append(stage1_map_point)
            point_list.append(stage2_map_point)

        draw_graph(cfg, epoch_point, point_list, fuse_map_point)

        # Train Pseudo
        
        if e >= 0.2 * cfg.real_total_epochs:   
            print('----------------------------------------------------')
            print('---------------   Train Pseudo  --------------------')
            print('----------------------------------------------------')

            self_paced = 1 * e / cfg.real_total_epochs

            pseudo_dataset = obj_from_dict(cfg.data2.train, datasets, dict(pseudo_set=pseudo_data_dict, self_paced=self_paced, pseudo_test=True, class_balanced=False))

            train_detector(
                model,
                pseudo_dataset,
                cfg,
                distributed=distributed,
                validate=args.validate,
                logger=logger, epoch=e, pseudo_dataset=True)


if __name__ == '__main__':
    main()
