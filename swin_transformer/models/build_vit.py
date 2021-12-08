
def build_model(cfg):
    if cfg.IS_CONSISTENT:
        from .vit_consistent import (
                vit_b_16_224,
                vit_b_16_384,
                vit_b_32_224,
                vit_b_32_384,
                vit_l_16_384,
                vit_l_32_384
                )
    else:
        from .vit_ddp import (
                vit_b_16_224,
                vit_b_16_384,
                vit_b_32_224,
                vit_b_32_384,
                vit_l_16_384,
                vit_l_32_384
                )
    if cfg.MODEL_ARCH == "vit_b_16_224":
        return vit_b_16_224()
    elif cfg.MODEL_ARCH == "vit_b_16_384":
        return vit_b_16_384()
    elif cfg.MODEL_ARCH == "vit_b_32_224":
        return vit_b_32_224()
    elif cfg.MODEL_ARCH == "vit_b_32_384":
        return vit_b_32_384()
    elif cfg.MODEL_ARCH == "vit_l_16_384":
        return vit_l_16_384()
    elif cfg.MODEL_ARCH == "vit_l_32_384":
        return vit_l_32_384()

