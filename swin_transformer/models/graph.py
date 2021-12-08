import oneflow as flow


class TrainGraph(flow.nn.Graph):
    def __init__(self, model, loss_fn, optimizer, lr_scheduler, accumulation_steps):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.accumulation_steps = accumulation_steps

    def build(self, image, label):
        outputs = self.model(image)
        loss = self.loss_fn(outputs, label)
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps

        loss.backward()
        return loss


class EvalGraph(flow.nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, image):
        outputs = self.model(image)
        return outputs

