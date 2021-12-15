import argparse
import enum
import os
import math
import matplotlib

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



def of_train(model, args):
    # # 载入一样的权重
    # model = load_from_torch(model, "./torch_weight.pth")

    model.cuda()
    model.train()
    loss_fn = flow_nn.CrossEntropyLoss()
    data_loader = of_dataloader(
        data_dir=args.data_path,
        crop_pct=0.875,
        batch_size=args.batch_size,
        num_workers=8,
        split="train",
    )
    optimizer = flow.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=0.001, weight_decay=0.05)
    optimizer.zero_grad()

    print("Start OneFlow Training")
    print("======================")

    with open("./loss_file/oneflow_loss.txt", "w") as f:
        for iter_idx, (data, target) in enumerate(data_loader):
            if iter_idx == args.total_iters:
                break
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()

            output = model(data)
            loss = loss_fn(output, target)
            print("iter: %d, loss: %.4f" % (iter_idx, loss.numpy()))
            f.write(str(loss.numpy()))
            f.write("\n")
            loss.backward()
            optimizer.step()


def torch_train(model, args):
    # # 载入一样初始化的权重
    # model.load_state_dict(torch.load("./torch_weight.pth"))

    model.cuda()
    model.train()
    loss_fn = torch_nn.CrossEntropyLoss()
    data_loader = torch_dataloader(
        data_dir=args.data_path,
        crop_pct=0.875,
        batch_size=args.batch_size,
        num_workers=8,
        split="train",
    )
    # parameters = set_weight_decay(model, {'absolute_pos_embed'}, {'relative_position_bias_table'})
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=0.001, weight_decay=0.05)
    optimizer.zero_grad()

    print("Start Pytorch Training")
    print("======================")

    with open("./loss_file/torch_loss.txt", "w") as f:
        for iter_idx, (data, target) in enumerate(data_loader):
            if iter_idx == args.total_iters:
                break
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()

            output = model(data)
            loss = loss_fn(output, target)
            print_loss = loss.cpu().detach().numpy()
            print("iter: %d, loss: %.4f" % (iter_idx, print_loss))
            f.write(str(print_loss))
            f.write("\n")
            loss.backward()
            optimizer.step()


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

    of_train(of_model, args)
    torch_train(torch_model, args)

    if args.draw:
        draw(of_file="./loss_file/oneflow_loss.txt", torch_file="./loss_file/torch_loss.txt")
