# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import oneflow as flow


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

