import oneflow as flow
import oneflow.nn as nn


def make_static_grad_scaler():
    return flow.amp.StaticGradScaler(flow.env.get_world_size())


def make_grad_scaler():
    return flow.amp.GradScaler(
        init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
    )


def meter(self, mkey, *args):
    assert mkey in self.m
    self.m[mkey]["meter"].record(*args)


class TrainGraph(flow.nn.Graph):
    def __init__(
        self,
        model,
        criterion,
        data_loader,
        optimizer,
        lr_scheduler,
        mixup_fn=None,
    ):
        super().__init__()

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)

        self.model = model
        self.criterion = criterion

        self.data_loader = data_loader
        self.mixup_fn = mixup_fn
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self):
        samples, targets = self.data_loader()
        samples = samples.to("cuda")
        targets = targets.to("cuda")
        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)
        outputs = self.model(samples.to_consistent(
                  placement=flow.placement("cuda", {0: range(flow.env.get_world_size())}), sbp=flow.sbp.split(0)))
        loss = self.criterion(outputs, targets)

        loss.backward()
        return loss


class EvalGraph(flow.nn.Graph):
    def __init__(self, model, cfg):
        super().__init__()
        self.config.allow_fuse_add_to_output(True)
        self.model = model

    def build(self, image):
        logits = self.model(image)
        return logits
