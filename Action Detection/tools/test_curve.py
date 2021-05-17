import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors

import os,errno
import os.path as osp
import scipy.io as sio

import matplotlib
from matplotlib.offsetbox import AnchoredText
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--ep', default=0, type=int, help='eps')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--small', action='store_true', help='show results')
    parser.add_argument('--trim', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True, small=args.small, trim=args.trim))
    
    eps = args.ep
    rgb_map_point = []
    flow_map_point = []

    stage1_map_point = []
    stage2_map_point = []

    for i in range(eps):
        print("epoch:",i)
        epoch_point = [(n+1) for n in range(i+1)]

        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint+'/'+'epoch_'+str(i+1)+'.pth')
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, args.show)
    
        eval_list = dataset.eval(outputs) #[0,0,0,0]#

        print("eval_list",eval_list)

        rgb_map_point.append(float("%.4f" % eval_list[0]))
        flow_map_point.append(float("%.4f" % eval_list[1]))
        if len(eval_list) > 2:
            stage1_map_point.append(float("%.4f" % eval_list[2]))
            stage2_map_point.append(float("%.4f" % eval_list[3]))

        # eval_types = args.eval

        # if eval_types:
        #     print('Starting evaluate {}'.format(' and '.join(eval_types)))
        #     if eval_types == ['proposal_fast']:
        #         result_file = args.out
        #     else:
        #         result_file = args.out + '.json'
        #         results2json(dataset, outputs, result_file)
        #     coco_eval(result_file, eval_types, dataset.coco)

        ######### draw ###########

        try:
            os.makedirs(args.checkpoint + '/graph/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        fig, ax = plt.subplots()

        ax.plot(epoch_point, rgb_map_point, 'k', label='RGB MAP',color='r')
        ax.plot(epoch_point, flow_map_point, 'k', label='Flow MAP',color='g')
        if len(eval_list) > 2:
            ax.plot(epoch_point, stage1_map_point, 'k', label='Stage1 MAP',color='b')
            ax.plot(epoch_point, stage2_map_point, 'k', label='Stage2 MAP',color='c')

        ax.annotate('Last Epoch Accuracy: %0.4f' % (rgb_map_point[-1]), xy=(1.05, 0.1), xycoords='axes fraction', size=14)

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

        plt.savefig(args.checkpoint + '/graph/map.png', bbox_inches='tight')
        
        fig.clf()

        plt.clf()

if __name__ == '__main__':
    main()
