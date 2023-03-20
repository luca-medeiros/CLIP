from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.vision_transformer import VisionTransformer

from clip.models.layers import get_conv_layer, get_conv_modules, get_norm_layer
from clip.models.utils import get_valid_args

__all__ = [
    'VisionTransformer', 'vit_b_p16_224x224', 'vit_b_p16_384x384', 'vit_b_p32_224x224', 'vit_l_p16_224x224',
    'vit_l_p16_512x512', 'vit_l_p32_224x224', 'vit_h_p14_224x224', 'vit_h_p14_518x518'
]

model_urls = {
    'vit_b_p16_224x224': 'https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth',
    'vit_b_p16_384x384': 'https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth',
    'vit_b_p32_224x224': 'https://download.pytorch.org/models/vit_b_32-d86f8d99.pth',
    'vit_l_p16_224x224': 'https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth',
    'vit_l_p16_512x512': 'https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth',
    'vit_l_p32_224x224': 'https://download.pytorch.org/models/vit_l_32-c7638314.pth',
    'vit_h_p14_224x224': 'https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth',
    'vit_h_p14_518x518': 'https://download.pytorch.org/models/vit_h_14_swag-80465313.pth',
}


class VisionTransformer(VisionTransformer):

    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 num_layers: int,
                 num_heads: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 inp_channels: int = 3,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 num_classes: int = 1000,
                 representation_size: Optional[int] = None,
                 norm_cfg: Optional[Dict[str, Any]] = None,
                 conv_module_cfgs: Optional[List[Dict[str, Any]]] = None,
                 weight: Optional[Dict[str, torch.Tensor]] = None) -> None:

        if norm_cfg is None:
            norm_layer = get_norm_layer({"name": 'ln', "eps": 1e-6})
        else:
            norm_layer = get_norm_layer(norm_cfg)

        super().__init__(
            image_size,
            patch_size,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            num_classes,
            representation_size,
            norm_layer,
            conv_stem_configs=None,
        )

        if conv_module_cfgs is not None:
            seq_proj = nn.Sequential()
            prev_channels = inp_channels

            for i, conv_module_cfg in enumerate(conv_module_cfgs):
                seq_proj.add_module(f"conv_bn_relu_{i}", get_conv_modules(*conv_module_cfg)(prev_channels))
                prev_channels = conv_module_cfg['conv_cfg']['out_chans']
        else:
            self.conv_proj = get_conv_layer({
                "name": 'conv',
                "kernel_size": patch_size,
                "stride": patch_size
            })(inp_channels, hidden_dim)

        if weight is not None:
            self._load_weight(weight)
            # TODO: logging
            # print_desc(f"[bold yellow][Inform] [white]model successfully loaded.")

    def _load_weight(self, state_dict: Dict[str, torch.Tensor]):
        matched_w, unmatched_w = OrderedDict(), OrderedDict()

        params = self.state_dict()
        for key in state_dict:
            if 'mlp' in key:
                matched_w[key] = state_dict[key]
                continue

            from_w = state_dict[key]
            from_shape = from_w.size()
            to_w = params.get(key, None)

            if to_w is None:
                unmatched_w[key] = f" unloaded. Cuz of unmatched name to {key}"
                continue
            elif from_shape != to_w.shape:
                unmatched_w[key] = f" unloaded. Cuz of different shape {from_shape} <-> {to_w.shape}"
                continue
            if from_w.size() == to_w.size():
                matched_w[key] = from_w

        # TODO: logging
        # for key, value in unmatched_w.items():
        #     print_desc(f"[bold red][Warning] [white]{key}" + value)

        self.load_state_dict(matched_w, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if hasattr(self, 'heads'):
            x = self.heads(x)
        return x


def vit_b_p16_224x224(pretrained=True, **kwargs):
    weight = None
    if pretrained:
        weight = model_zoo.load_url(model_urls['vit_b_p16_224x224'])

    norm_cfg = kwargs.get('norm_cfg', {"type": "ln", "eps": 1e-6})
    args = {
        "image_size": 224,
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "norm_cfg": norm_cfg
    }
    if kwargs is not None:
        valid_args = get_valid_args(VisionTransformer, kwargs)
        args.update(valid_args)

    return VisionTransformer(**args, weight=weight)


def vit_b_p32_224x224(pretrained=True, **kwargs):
    weight = None
    if pretrained:
        weight = model_zoo.load_url(model_urls['vit_b_p32_224x224'])

    norm_cfg = kwargs.get('norm_cfg', {"type": "ln", "eps": 1e-6})
    args = {
        "image_size": 224,
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "norm_cfg": norm_cfg
    }
    if kwargs is not None:
        valid_args = get_valid_args(VisionTransformer, kwargs)
        args.update(valid_args)

    return VisionTransformer(**args, weight=weight)


def vit_b_p16_384x384(pretrained=True, **kwargs):
    weight = None
    if pretrained:
        weight = model_zoo.load_url(model_urls['vit_b_p16_384x384'])

    norm_cfg = kwargs.get('norm_cfg', {"type": "ln", "eps": 1e-6})
    args = {
        "image_size": 384,
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "norm_cfg": norm_cfg
    }
    if kwargs is not None:
        valid_args = get_valid_args(VisionTransformer, kwargs)
        args.update(valid_args)

    return VisionTransformer(**args, weight=weight)


def vit_l_p16_224x224(pretrained=True, **kwargs):
    weight = None
    if pretrained:
        weight = model_zoo.load_url(model_urls['vit_l_p16_224x224'])

    norm_cfg = kwargs.get('norm_cfg', {"type": "ln", "eps": 1e-6})
    args = {
        "image_size": 224,
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "norm_cfg": norm_cfg
    }
    if kwargs is not None:
        valid_args = get_valid_args(VisionTransformer, kwargs)
        args.update(valid_args)

    return VisionTransformer(**args, weight=weight)


def vit_l_p32_224x224(pretrained=True, **kwargs):
    weight = None
    if pretrained:
        weight = model_zoo.load_url(model_urls['vit_l_p32_224x224'])

    norm_cfg = kwargs.get('norm_cfg', {"type": "ln", "eps": 1e-6})
    args = {
        "image_size": 224,
        "patch_size": 32,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "norm_cfg": norm_cfg
    }
    if kwargs is not None:
        valid_args = get_valid_args(VisionTransformer, kwargs)
        args.update(valid_args)

    return VisionTransformer(**args, weight=weight)


def vit_l_p16_512x512(pretrained=True, **kwargs):
    weight = None
    if pretrained:
        weight = model_zoo.load_url(model_urls['vit_l_p16_512x512'])

    norm_cfg = kwargs.get('norm_cfg', {"type": "ln", "eps": 1e-6})
    args = {
        "image_size": 512,
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "norm_cfg": norm_cfg
    }
    if kwargs is not None:
        valid_args = get_valid_args(VisionTransformer, kwargs)
        args.update(valid_args)

    return VisionTransformer(**args, weight=weight)


def vit_h_p14_224x224(pretrained=True, **kwargs):
    weight = None
    if pretrained:
        weight = model_zoo.load_url(model_urls['vit_l_p16_512x512'])

    norm_cfg = kwargs.get('norm_cfg', {"type": "ln", "eps": 1e-6})
    args = {
        "image_size": 224,
        "patch_size": 14,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "norm_cfg": norm_cfg
    }
    if kwargs is not None:
        valid_args = get_valid_args(VisionTransformer, kwargs)
        args.update(valid_args)

    return VisionTransformer(**args, weight=weight)


def vit_h_p14_518x518(pretrained=True, **kwargs):
    weight = None
    if pretrained:
        weight = model_zoo.load_url(model_urls['vit_h_p14_518x518'])

    norm_cfg = kwargs.get('norm_cfg', {"type": "ln", "eps": 1e-6})
    args = {
        "image_size": 518,
        "patch_size": 14,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "norm_cfg": norm_cfg
    }
    if kwargs is not None:
        valid_args = get_valid_args(VisionTransformer, kwargs)
        args.update(valid_args)

    return VisionTransformer(**args, weight=weight)
