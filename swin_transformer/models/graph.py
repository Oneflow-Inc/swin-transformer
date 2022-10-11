import oneflow as flow

def make_grad_scaler():
    # return flow.amp.GradScaler(
    #     init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
    # )
    return flow.amp.GradScaler(
        init_scale=65536.0 * flow.env.get_world_size(),
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )

def make_static_grad_scaler():
    return flow.amp.StaticGradScaler(flow.env.get_world_size())

class TrainGraph(flow.nn.Graph):
    def __init__(self, model, loss_fn, optimizer, lr_scheduler):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

        # if args.use_fp16:
        # self.config.enable_amp(True)
        # self.set_grad_scaler(make_grad_scaler())
        # elif args.scale_grad:
        self.set_grad_scaler(make_static_grad_scaler())

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_cast_scale(True)

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
        # self.config.enable_amp(True)

    def build(self, image):
        outputs = self.model(image)
        return outputs
