import oneflow as flow


class TrainGraph(flow.nn.Graph):
    def __init__(self, model, loss_fn, optimizer, train_data_loader, lr_scheduler, mixup_fn, accumulation_steps):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.train_data_loader = train_data_loader
        self.mixup_fn = mixup_fn
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.accumulation_steps = accumulation_steps

    def build(self):
        image, label = self.train_data_loader
        image = image.to("cuda")
        label = label.to("cuda")
        label = label.to_consistent(
            placement=flow.placement("cuda", {0: range(flow.env.get_world_size())}), sbp=flow.sbp.split(0))
        if self.mixup_fn is not None:
            image, label = self.mixup_fn(image, label)
        outputs = self.model(image.to_consistent(
            placement=flow.placement("cuda", {0: range(flow.env.get_world_size())}), sbp=flow.sbp.split(0)))
        loss = self.loss_fn(outputs, label)
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps

        loss.backward()
        return loss


class EvalGraph(flow.nn.Graph):
    def __init__(self, model, val_data_loader):
        super().__init__()
        self.model = model
        self.val_data_loader = val_data_loader

    def build(self):
        image, label = self.val_data_loader
        image = image.to("cuda")
        label = label.to("cuda")
        label = label.to_consistent(
            placement=flow.placement("cuda", {0: range(flow.env.get_world_size())}), sbp=flow.sbp.split(0))
        outputs = self.model(image.to_consistent(
            placement=flow.placement("cuda", {0: range(flow.env.get_world_size())}), sbp=flow.sbp.split(0)))
        return outputs, label

