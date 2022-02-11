import oneflow as flow
from oneflow import optim as optim

from flowvision.scheduler.cosine_lr import CosineLRScheduler
from flowvision.scheduler.step_lr import StepLRScheduler
from flowvision.scheduler.scheduler import Scheduler

def build_optimizer(config, model, graph_mode=False):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    # FIXME: weight decay on pos embed
    parameters = set_weight_decay(model, {'absolute_pos_embed'}, {'relative_position_bias_table'})

    if config.TRAIN.CLIP_GRAD == 1.0:
        for param_group in parameters:
            param_group["clip_grad_max_norm"] = (1.0,)
            param_group["clip_grad_norm_type"] = (2.0,)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)


    lr_scheduler = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=num_steps
    )

    lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
        lr_scheduler,
        warmup_factor=config.TRAIN.WARMUP_LR,
        warmup_iters=warmup_steps,
        warmup_method="linear",
    )
    return lr_scheduler

def make_grad_scaler():
    return flow.amp.GradScaler(
        init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
    )

def make_static_grad_scaler():
    return flow.amp.StaticGradScaler(flow.env.get_world_size())

class TrainGraph(flow.nn.Graph):
    def __init__(self, model, loss_fn, optimizer, lr_scheduler):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

        # if args.use_fp16:
        #     self.config.enable_amp(True)
        #     self.set_grad_scaler(make_grad_scaler())
        # elif args.scale_grad:
        self.set_grad_scaler(make_static_grad_scaler())

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)

        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self, image, label):
        outputs = self.model(image)
        loss = self.loss_fn(outputs, label)
        loss.backward()
        return loss

class EvalGraph(flow.nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, image):
        outputs = self.model(image)
        return outputs
