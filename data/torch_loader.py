import os
import math

# oneflow impl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class ImageNetDataLoader(DataLoader):
    def __init__(
        self,
        data_dir,
        split="train",
        image_size=224,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        crop_pct=0.875,
        batch_size=16,
        num_workers=8,
    ):

        if split == "train":
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(img_mean, img_std),
                ]
            )
        else:
            scale_size = int(math.floor(image_size / crop_pct))
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        scale_size, interpolation=3
                    )  # 3: bibubic
                    if image_size == 224
                    else transforms.Resize(image_size, interpolation=3),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(img_mean, img_std),
                ]
            )

        self.dataset = ImageFolder(
            root=os.path.join(data_dir, split), transform=transform
        )
        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )