# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime

import numpy as np

import oneflow as flow
import oneflow.profiler as  profiler

from flowvision.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from flowvision.utils.metrics import accuracy, AverageMeter

from config import get_config
from models import build_model
from models.graph import TrainGraph, EvalGraph
from data import build_loader
from optimizer import build_optimizer
from lr_scheduler import build_scheduler

from libai.utils import distributed as dist
from libai.config import LazyConfig

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--config_file', type=str, help='path to dataset')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

if __name__ == '__main__':
    args, config = parse_option()

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, ["train.dist.tensor_parallel_size=1"])
    dist.setup_dist_util(cfg.train.dist)

    flow.boxing.nccl.set_fusion_threshold_mbytes(16)
    flow.boxing.nccl.set_fusion_max_ops_num(24)

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    model = build_model(config)

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = flow.nn.CrossEntropyLoss()

    # placement = flow.placement("cuda", {0: [i for i in range(flow.env.get_world_size())]}, (2, 4),)
    # sbp = [flow.sbp.broadcast, flow.sbp.broadcast]
    # placement = flow.env.all_device_placement("cuda")
    # sbp = flow.sbp.broadcast
    
    # model.to_consistent(placement=dist.get_layer_placement(0), sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]))
    optimizer = build_optimizer(config, model)
    lr_scheduler = flow.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)

    train_graph = TrainGraph(model=model,
                             loss_fn=criterion,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler)

    data_loader_train_iter = iter(data_loader_train)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    max_accuracy = 0.0
    start_time = time.time()
    end = time.time()

    # sbp = [flow.sbp.split(0), flow.sbp.split(0)]
    sbp = flow.sbp.split(0)

    for idx in range(200):
        model.train()
        optimizer.zero_grad()

        samples, targets = data_loader_train_iter.__next__()
        samples = samples.cuda()
        targets = targets.cuda()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        
        samples = samples.to_consistent(placement=dist.get_layer_placement(0), sbp=sbp)
        targets = targets.to_consistent(placement=dist.get_layer_placement(0), sbp=sbp)

        loss = train_graph(samples, targets)

        batch_time.update(time.time() - end)
        end = time.time()

    local_tensor = loss.to_local().numpy()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if flow.env.get_rank() == 0:
        print(total_time_str)




