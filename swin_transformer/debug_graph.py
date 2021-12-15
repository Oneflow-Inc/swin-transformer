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

# import oneflow.backends.cudnn as cudnn

from flowvision.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from flowvision.utils.metrics import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
# from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor


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

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

class EvalGraph(flow.nn.Graph):
    def __init__(self, model):
        super().__init__()

        # if args.use_fp16:
        #     self.config.enable_amp(True)
        # self.config.allow_fuse_add_to_output(True)
        self.model = model

    def build(self, image):
        # image = image.to("cuda")
        logits = self.model(image)
        return logits
        # pred = logits.softmax()
        # return pred

def build_graph():
    return None

if __name__ == '__main__':
    args, config = parse_option()
    # dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    
    model = build_model(config)
    model.cuda()

    # model.to_consistent(sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cuda"))
    # eval_graph = EvalGraph(model)

    input_tensor = flow.ones(config.DATA.BATCH_SIZE, 3, 224, 224, dtype=flow.float32,
                             sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cuda"))

    output = model(input_tensor)

    # output = eval_graph(input_tensor)



    # optimizer = build_optimizer(config, model)
    # model = flow.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
    # model_without_ddp = model

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"number of params: {n_parameters}")
    # if hasattr(model_without_ddp, 'flops'):
    #     flops = model_without_ddp.flops()
    #     logger.info(f"number of GFLOPs: {flops / 1e9}")

    # lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    # # lr_scheduler = flow.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = flow.nn.CrossEntropyLoss()





