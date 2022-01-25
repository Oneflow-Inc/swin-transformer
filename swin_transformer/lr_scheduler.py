# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import oneflow as torch
from flowvision.scheduler.cosine_lr import CosineLRScheduler
from flowvision.scheduler.step_lr import StepLRScheduler
from flowvision.scheduler.scheduler import Scheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)


    lr_scheduler = torch.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=decay_steps
    )

    lr_scheduler = torch.optim.lr_scheduler.WarmUpLR(
        lr_scheduler,
        warmup_factor=0,
        warmup_iters=warmup_steps,
        warmup_method="linear",
    )
    return lr_scheduler
