from __future__ import division

from collections import OrderedDict

import torch
from mmcv.runner import Runner, DistSamplerSeedHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.core import (DistOptimizerHook, CocoDistEvalRecallHook,
                        CocoDistEvalmAPHook)
from mmdet.datasets import build_dataloader
from mmdet.models import RPN
from .env import get_root_logger


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None,epoch=0,pseudo_dataset=False):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate,epoch=epoch)
    else:
        rgb_acc, flow_acc = _non_dist_train(model, dataset, cfg, validate=validate,epoch=epoch, pseudo_dataset=pseudo_dataset)
        return rgb_acc, flow_acc

def _dist_train(model, dataset, cfg, validate=False,epoch=0):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True,ucfdiv=cfg.ucfdiv)
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        if isinstance(model.module, RPN):
            runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
        elif cfg.data.val.type == 'CocoDataset':
            runner.register_hook(CocoDistEvalmAPHook(cfg.data.val))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False,epoch=0, pseudo_dataset=False):
    # print('Epoch {}:'.format(epoch))

    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False,set1_len=dataset.set1_len,set2_len=dataset.set2_len)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    if epoch == 0:
        use_lr_config = cfg.lr_config
        optim = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
        if pseudo_dataset:
            load_from = cfg.load_from
        else:
            load_from = None
    else:
        p = float(epoch) / cfg.real_total_epochs
        lr_p = (1 + 10 * p)**(-0.75)
        use_lr_config = dict(policy='step', step=[99999999])
        optim = dict(type='SGD', lr=0.01*lr_p, momentum=0.9, weight_decay=0.0001)
        load_from = cfg.load_from

    # build runner
    runner = Runner(model, batch_processor, optim, cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(use_lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # if cfg.resume_from:
    #     runner.resume(cfg.resume_from)
    if load_from is not None:
        runner.load_checkpoint(load_from)

    # print("current_lr ", runner.current_lr())
    rgb_acc, flow_acc = runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    print("rgb_acc, flow_acc",rgb_acc, flow_acc)

    return rgb_acc, flow_acc
