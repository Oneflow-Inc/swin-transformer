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
import oneflow.backends.cudnn as cudnn

from flowvision.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from flowvision.utils.metrics import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor


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
    parser.add_argument('--batch-size', type=int, default=8, help="batch size for single GPU")
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

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, required=False, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model_without_ddp = model
    logger.info(str(model))
    
    # placement = flow.placement("cuda", {0: [i for i in range(flow.env.get_world_size())]}, (2, 4),)
    # sbp = [flow.sbp.broadcast, flow.sbp.broadcast]
    placement = flow.env.all_device_placement("cuda")
    sbp = flow.sbp.broadcast
    model.to_consistent(placement=placement, sbp=sbp)

    optimizer = build_optimizer(config, model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    # criterion = flow.nn.CrossEntropyLoss()

    max_accuracy = 0.0

        # acc1, acc5, loss = validate(config, data_loader_val, model)
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # if config.EVAL_MODE:
        #     return

    # if config.THROUGHPUT_MODE:
    #     throughput(data_loader_val, model, logger)
    #     return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        # if flow.env.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        # no validate
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    # placement = flow.placement("cuda", {0: [i for i in range(flow.env.get_world_size())]}, (2, 4),)
    # split_sbp = [flow.sbp.split(0), flow.sbp.split(0)]
    # broadcast_sbp = [flow.sbp.broadcast, flow.sbp.broadcast]
    placement = flow.env.all_device_placement("cuda")
    split_sbp = flow.sbp.split(0)
    broadcast_sbp = flow.sbp.broadcast

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda()
        targets = targets.cuda()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        samples = samples.to_consistent(placement=placement, sbp=split_sbp)
        targets = targets.to_consistent(placement=placement, sbp=split_sbp)

        outputs = model(samples)

        loss = criterion(outputs, targets)
        loss = loss * flow.env.get_world_size()

        optimizer.zero_grad()
        loss.backward()

        for param_group in optimizer.param_groups:
            for param in param_group.parameters:
                param.grad.detach().mul_( 1.0 / flow.env.get_world_size())

        # if config.TRAIN.CLIP_GRAD:
        grad_norm = flow.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        # else:
        # grad_norm = get_grad_norm(model.parameters())
 
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        loss = loss.to_consistent(sbp=broadcast_sbp).to_local()
        grad_norm = grad_norm.to_local()
        loss_meter.update(loss.item() / flow.env.get_world_size(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@flow.no_grad()
def validate(config, data_loader, model):
    criterion = flow.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    # placement = flow.placement("cuda", {0: [i for i in range(flow.env.get_world_size())]}, (2, 4),)
    # split_sbp = [flow.sbp.split(0), flow.sbp.split(0)]
    # broadcast_sbp = [flow.sbp.broadcast, flow.sbp.broadcast]
    placement = flow.env.all_device_placement("cuda")
    split_sbp = flow.sbp.split(0)
    broadcast_sbp = flow.sbp.broadcast

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        # images = images.cuda()
        # target = target.cuda()
        images = images.to_consistent(placement=placement, sbp=split_sbp)
        target = target.to_consistent(placement=placement, sbp=split_sbp)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        output = output.to_consistent(sbp=broadcast_sbp).to_local()
        target = target.to_consistent(sbp=broadcast_sbp).to_local()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # acc1 = #reduce_tensor(acc1)
        # acc5 = #reduce_tensor(acc5)
        loss = loss.to_consistent(sbp=broadcast_sbp).to_local() #reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t')

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@flow.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda()
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        # flow.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        # flow.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    # import multiprocessing as mp
    # mp.set_start_method("spawn")

    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = flow.env.get_rank()
        world_size = flow.env.get_world_size()
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    seed = config.SEED + flow.env.get_rank()
    flow.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * flow.env.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * flow.env.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * flow.env.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=flow.env.get_rank(), name=f"{config.MODEL.NAME}")

    if flow.env.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
