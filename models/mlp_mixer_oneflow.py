import math
from functools import partial
import warnings

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.init as init
from oneflow.nn.init import _calculate_fan_in_and_fan_out

from typing import Any, Callable, Optional, Tuple

def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

# helpers
def to_2tuple(x):
    return (x, x)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0]
        ), f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (
            W == self.img_size[1]
        ), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x



def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        init.trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


# helpers
def pair(x):
    if not isinstance(x, tuple):
        return (x, x)
    else:
        return x


class Mlp(nn.Module):
    """
    You can also import Mlp Block in flowvision.layers.blocks like this:
    from flowvision.layers.blocks import Mlp
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        gate_layer=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit
    """

    def __init__(self, dim, num_patches, norm_layer=nn.LayerNorm):
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(num_patches, num_patches)

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        # TODO: use fixed chunk op
        # u, v = x.chunk(2, dim=-1)
        B, N, C = x.size()
        split_dim = C // 2
        u, v = flow.split(x, split_dim, dim=-1)[0], flow.split(x, split_dim, dim=-1)[1]
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating
    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(
        self,
        dim,
        num_patches,
        mlp_ratio=4,
        mlp_layer=GatedMlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        sgu = partial(SpatialGatingUnit, num_patches=num_patches)
        self.mlp_channels = mlp_layer(
            dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """

    def __init__(
        self,
        dim,
        num_patches,
        mlp_ratio=(0.5, 4.0),
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in pair(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(
            num_patches, tokens_dim, act_layer=act_layer, drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(
            self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        )
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class MlpMixer(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        img_size=224,
        in_chans=3,
        patch_size=16,
        num_blocks=8,
        embed_dim=512,
        mlp_ratio=(0.5, 4.0),
        block_layer=MixerBlock,
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop_rate=0.0,
        drop_path_rate=0.0,
        nlhb=False,
        stem_norm=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        self.stem = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if stem_norm else None,
        )
        # TODO consistent the drop-path-rate rule with the original repo
        self.blocks = nn.Sequential(
            *[
                block_layer(
                    embed_dim,
                    self.stem.num_patches,
                    mlp_ratio,
                    mlp_layer=mlp_layer,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=drop_path_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        )

        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.0
        named_apply(
            partial(_init_weights, head_bias=head_bias), module=self
        )  # depth-first

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_weights(module: nn.Module, name: str, head_bias: float = 0.0, flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if "mlp" in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def _create_mlp_mixer(arch, pretrained=False, progress=True, **model_kwargs):
    model = MlpMixer(**model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def mlp_mixer_s16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-S/16 224x224 model.
    .. note::
        Mixer-S/16 224x224 model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> mlp_mixer_s16_224 = flowvision.models.mlp_mixer_s16_224(pretrained=False, progress=True)
    """
    model_kwargs = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    return _create_mlp_mixer(
        "mlp_mixer_s16_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


def mlp_mixer_s32_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-S/32 224x224 model.
    .. note::
        Mixer-S/32 224x224 model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> mlp_mixer_s32_224 = flowvision.models.mlp_mixer_s32_224(pretrained=False, progress=True)
    """
    model_kwargs = dict(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    return _create_mlp_mixer(
        "mlp_mixer_s32_224", pretrained=pretrained, progress=progress, **model_kwargs
    )


def mlp_mixer_b16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the Mixer-B/16 224x224 model.
    .. note::
        Mixer-B/16 224x224 model from `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/pdf/2105.01601.pdf>`_.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderrt. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> mlp_mixer_b16_224 = flowvision.models.mlp_mixer_b16_224(pretrained=False, progress=True)
    """
    model_kwargs = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    return _create_mlp_mixer(
        "mlp_mixer_b16_224", pretrained=pretrained, progress=progress, **model_kwargs
    )
