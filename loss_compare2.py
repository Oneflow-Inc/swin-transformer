import argparse
import enum
import os
import math
import matplotlib
from torch._C import dtype

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# oneflow impl
import oneflow as flow
import oneflow.nn as flow_nn
from data.of_loader import ImageNetDataLoader as of_dataloader
from models.swin_oneflow import SwinTransformer as of_swin

# torch impl
import torch
import torch.nn as torch_nn
from data.torch_loader import ImageNetDataLoader as torch_dataloader
from models.swin_pytorch import SwinTransformer as torch_swin

from utils import *



def train(of_model, torch_model, args):
    # oneflow setup
    of_model.cuda()
    of_model.train()
    of_loss = flow_nn.CrossEntropyLoss()
    # of_optim = flow.optim.AdamW(of_model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=0.001, weight_decay=0.05)
    of_optim = flow.optim.SGD(of_model.parameters(), lr=0.001, momentum=0.9)
    of_optim.zero_grad()

    # torch setup
    torch_model.cuda()
    torch_model.train()
    torch_loss = torch_nn.CrossEntropyLoss()
    # torch_optim = torch.optim.AdamW(torch_model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=0.001, weight_decay=0.05)
    torch_optim = torch.optim.SGD(torch_model.parameters(), lr=0.001, momentum=0.9)
    torch_optim.zero_grad()

    # data setup
    data_loader = torch_dataloader(
        data_dir=args.data_path,
        crop_pct=0.875,
        batch_size=args.batch_size,
        num_workers=8,
        split="train",
    )

    print("Start Training")
    print("==============")

    for iter_idx, (data, target) in enumerate(data_loader):
        if iter_idx == args.total_iters:
            break
        # train torch model
        torch_optim.zero_grad()
        data = data.cuda()
        target = target.cuda()
        output = torch_model(data)
        th_loss = torch_loss(output, target)
        th_print_loss = th_loss.cpu().detach().numpy()
        print("iter: %d, pytorch loss: %.4f" % (iter_idx, th_print_loss))
        # using w+
        with open("./loss_file/torch_loss.txt", "a") as f:
            f.write(str(th_print_loss))
            f.write("\n")
        th_loss.backward()
        torch_optim.step()
        
        # train oneflow model
        of_optim.zero_grad()
        of_data = flow.tensor(data.cpu().numpy(), dtype=flow.float32).cuda()
        of_target = flow.tensor(target.cpu().numpy()).long().cuda()
        of_output = of_model(of_data)
        flow_loss = of_loss(of_output, of_target)
        print("iter: %d, loss: %.4f" % (iter_idx, flow_loss.numpy()))
        with open("./loss_file/oneflow_loss.txt", "a") as f:
            f.write(str(flow_loss.numpy()))
            f.write("\n")
        flow_loss.backward()
        of_optim.step()


def draw(of_file, torch_file):
    of_loss = []
    torch_loss = []
    with open(of_file, "r") as f:
        for _line in f.readlines():
            of_loss.append(float(_line.strip()))
    
    with open(torch_file, "r") as f:
        for _line in f.readlines():
            torch_loss.append(float(_line.strip()))

    # setup
    plt.rcParams["figure.dpi"] = 100
    plt.clf()
    plt.xlabel("iter", fontproperties="Times New Roman")
    plt.ylabel("loss", fontproperties="Times New Roman")

    idx = [i for i in range(len(of_loss))]
    plt.plot(idx, of_loss, label="oneflow loss")
    plt.plot(idx, torch_loss, label="torch loss")
    plt.legend(loc="upper right", frameon=True, fontsize=8)
    plt.savefig("./loss_compare.png")



def _parse_args():
    parser = argparse.ArgumentParser("loss compare")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size",
    )
    parser.add_argument(
        "--data_path", type=str, default="/DATA/disk1/ImageNet/extract/", help="path to imagenet2012"
    )
    parser.add_argument(
        "--total_iters", type=int, default=100, help="total-iters"
    )
    parser.add_argument(
        "--draw", action="store_true", help="draw loss picture"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not os.path.exists("./loss_file"):
        os.mkdir("./loss_file")
    
    # init models 载入一样的权重
    torch_model = torch_swin(drop_path_rate=0.0, drop_rate=0.0, attn_drop_rate=0.0)
    of_model = of_swin(drop_path_rate=0.0, drop_rate=0.0, attn_drop_rate=0.0)
    of_model = load_from_torch(of_model, torch_model.state_dict())

    train(of_model, torch_model, args)

    if args.draw:
        draw(of_file="./loss_file/oneflow_loss.txt", torch_file="./loss_file/torch_loss.txt")
