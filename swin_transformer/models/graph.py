import oneflow as flow

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
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

        # Auto parallelism config
        # self.config.enable_auto_parallel(True)
        # self.config.enable_auto_parallel_mainstem_algo(True)
        # self.config.enable_auto_parallel_sbp_collector(True)
        # self.config.set_auto_parallel_computation_cost_ratio(0.25)
        # self.config.set_auto_parallel_wait_time(1.65e7)
        # self.config.set_auto_parallel_transfer_cost(1.65e7)

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
