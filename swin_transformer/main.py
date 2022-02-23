import argparse
import datetime
import functools
import logging
import numpy as np
import os
import sys
import time
from termcolor import colored

import oneflow as flow

from flowvision.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from flowvision.utils.metrics import accuracy, AverageMeter

from libai.config import LazyConfig
from libai.utils import distributed as dist

from config import get_config
from data import build_loader
from models.graph import TrainGraph, EvalGraph, build_optimizer, build_scheduler

from configs.swin_tiny_patch4_window7_224 import swin_tiny_patch4_window7_224_model
from libai.config import instantiate

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
    parser.add_argument('--libai_config_file', type=str, help='path to dataset')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def load_checkpoint(config, graph, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    checkpoint = flow.load(config.MODEL.RESUME, global_src_rank=0)
    msg = graph.load_state_dict(checkpoint['graph'], strict=True)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'epoch' in checkpoint:
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
    return max_accuracy


def save_checkpoint(config, epoch, graph, max_accuracy, logger):
    save_state = {'graph': graph.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'model_{epoch}')
    logger.info(f"{save_path} saving......")
    flow.save(save_state, save_path, global_dst_rank=0)
    logger.info(f"{save_path} saved !!!")


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    flow.boxing.nccl.set_fusion_threshold_mbytes(16)
    flow.boxing.nccl.set_fusion_max_ops_num(24)

    # model = build_model(config)
    model = instantiate(swin_tiny_patch4_window7_224_model)
    logger.info(str(model))

    model_without_ddp = model

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = flow.nn.CrossEntropyLoss()

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    train_graph = TrainGraph(model=model,
                             loss_fn=criterion,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler)
    eval_graph = EvalGraph(model=model)

    max_accuracy = 0.0
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, train_graph, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, train_graph, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, train_graph, max_accuracy, logger)

        # no validate
        acc1, acc5, loss = validate(config, data_loader_val, model, eval_graph)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, train_graph, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    
    placement = dist.get_layer_placement(0)
    input_sbp = dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.split(0)])
    loss_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda()
        targets = targets.cuda()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        samples = samples.to_global(placement=placement, sbp=input_sbp)
        targets = targets.to_global(placement=placement, sbp=input_sbp)

        loss = train_graph(samples, targets)

        loss_meter.update(loss.to_global(sbp=loss_sbp).to_local().item(), targets.size(0))
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
def validate(config, data_loader, model, eval_graph):
    model.eval()

    placement = dist.get_layer_placement(0)
    input_sbp = dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.split(0)])
    out_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])

    criterion = flow.nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.to_global(placement=placement, sbp=input_sbp)
        target = target.to_global(placement=placement, sbp=input_sbp)

        # compute output
        output = eval_graph(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        output = output.to_global(sbp=out_sbp).to_local()
        target = target.to_global(sbp=out_sbp).to_local()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss = loss.to_global(sbp=out_sbp).to_local()

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
def throughput(data_loader, eval_graph, logger):
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda()
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    cfg = LazyConfig.load(args.libai_config_file)
    dist.setup_dist_util(cfg.train.dist)

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
