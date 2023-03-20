from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import ops

from clip.models.utils import constant_init, get_valid_args, kaiming_init


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed. It contains non-trainable
    buffers called "weight" and "bias", "running_mean", "running_var", initialized to perform identity
    transformation. The pre-trained backbone models from Caffe2 only contain "weight" and "bias", which are
    computed from the original four parameters of BN.

    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    Other pre-trained backbone models may contain all 4 parameters.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def __repr__(self):
        return f"FrozenBatchNorm2d(num_features={self.num_features}, eps={self.eps})"

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class GroupNorm(nn.GroupNorm):

    def __init__(self,
                 num_channels: int,
                 num_groups: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 device: Any = None,
                 dtype: Any = None):
        super().__init__(num_groups, num_channels, eps, affine, device, dtype)


norms = {
    "ln": nn.LayerNorm,
    "bn": nn.BatchNorm2d,
    "sync": nn.SyncBatchNorm,
    "gn": GroupNorm,
    "frozen": FrozenBatchNorm2d,
}


def get_norm_layer(norm_cfg: Dict[str, Any]) -> Optional[Callable[[], nn.Module]]:
    args = deepcopy(norm_cfg)
    if norm_cfg is None:
        norm_cfg = {"type": None}
        args = {}

    name = norm_cfg.get('name', norm_cfg.get('type'))
    if name is not None:
        assert name.lower() in norms.keys()
        norm_layer = norms[name.lower()]
    else:
        return None

    if norm_cfg is not None:
        args = get_valid_args(norm_layer, args)
        norm_layer = partial(norm_layer, **args)
    return norm_layer


activations = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "relu6": nn.ReLU6,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "prelu": nn.PReLU,
    "silu": nn.SiLU,
    "hardswish": nn.Hardswish,
}


def get_act_layer(act_cfg: Dict[str, Any] = None) -> Optional[Callable[[], nn.Module]]:
    if act_cfg is None:
        act_cfg = {"type": None}

    name = act_cfg.get('name', act_cfg.get('type'))
    if name is not None:
        assert name.lower() in activations.keys()
        act_layer = activations[name.lower()]
    else:
        return None

    args = {}
    if act_cfg is not None:
        nargs = get_valid_args(act_layer, act_cfg)
        args.update(nargs)

    if name is not None and name.lower() == 'leaky_relu':
        if 'type' in act_cfg:
            act_cfg['type'] = 'leaky_relu'
        else:
            act_cfg['name'] = 'leaky_relu'
    act_layer = partial(act_layer, **args)
    return act_layer


class ConvModule(nn.Module):
    __constants__ = 'layers'

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            conv_cfg: Dict[str, Any] = None,
            norm_cfg: Dict[str, Any] = {"type": "BN"},
            act_cfg: Dict[str, Any] = {
                "type": 'Leaky_ReLU',
                "inplace": True,
                "negative_slope": 0.1
            },
            order: Tuple[str] = ("conv", "norm", "act"),
    ) -> None:
        super().__init__()

        assert conv_cfg is not None, "convolution layer config must given"

        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.in_chans = in_chans
        self.out_chans = out_chans

        if self.conv_cfg.get('padding', None) == 'auto':
            self.conv_cfg['padding'] = self.conv_cfg['kernel_size'] // 2
        if self.conv_cfg.get('bias', None) == "auto":
            self.conv_cfg['bias'] = False if self.norm_cfg is not None else True

        layers = self.set_layers()
        self.layers = nn.ModuleList()
        for o in order:
            layer = layers[o]
            if layer is None:
                continue
            self.layers.append(layer)
            self.add_module(o, layer)

    def set_layers(self) -> dict:
        layers = {}
        act_layer = get_act_layer(self.act_cfg)
        norm_layer = get_norm_layer(self.norm_cfg)
        conv_layer = get_conv_layer(self.conv_cfg)

        layers['conv'] = conv_layer(self.in_chans, self.out_chans)
        layers['norm'] = norm_layer(self.out_chans) if norm_layer is not None else None
        layers['act'] = act_layer() if act_layer is not None else None

        info = self.set_info(layers['conv'])
        for key, val in info.items():
            setattr(self, key, val)
        return layers

    def set_info(self, layer) -> dict:
        info = {}
        for key in [
                'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'transposed',
                'output_padding'
        ]:
            info[key] = getattr(layer, key)

        return info

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class DWConvModule(ConvModule):

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        conv_cfg: Dict[str, Any] = None,
        norm_cfg: Dict[str, Any] = {"type": "BN"},
        act_cfg: Dict[str, Any] = {
            "type": 'Leaky_ReLU',
            "inplace": True,
            "negative_slope": 0.1
        },
        order: Tuple[str] = ("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act")
    ) -> None:
        super().__init__(in_chans, out_chans, conv_cfg, norm_cfg, act_cfg, order=order)

    def set_layers(self) -> dict:
        layers = {}

        with_norm = self.norm_cfg is not None
        act_layer = get_act_layer(self.act_cfg)
        norm_layer = get_norm_layer(self.norm_cfg)
        depthwise_layer = get_conv_layer(self.conv_cfg)
        pointwise_layer = get_conv_layer({"name": 'conv', "kernel_size": 1, "stride": 1, "padding": 0})

        layers['act'] = act_layer() if act_layer is not None else None
        layers['dwnorm'] = norm_layer(self.in_chans) if norm_layer is not None else None
        layers['pwnorm'] = norm_layer(self.out_chans) if norm_layer is not None else None
        layers['depthwise'] = pointwise_layer(self.in_chans, self.in_chans, groups=self.in_chans)
        layers['pointwise'] = depthwise_layer(self.in_chans, self.out_chans)

        act_name = 'relu'
        if self.act_cfg is not None:
            act_name = self.act_cfg.get('type', self.act_cfg.get('name'))
            act_name = act_name.lower()

        kaiming_init(layers['depthwise'], nonlinearity=act_name)
        kaiming_init(layers['pointwise'], nonlinearity=act_name)
        if with_norm:
            constant_init(layers['dwnorm'], val=1, bias=0)
            constant_init(layers['pwnorm'], val=1, bias=0)

        info = self.set_info(layers['depthwise'])
        for key, val in info.items():
            setattr(self, key, val)

        return layers

    def set_info(self, layer) -> dict:
        info = {}
        for key in [
                'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'transposed',
                'output_padding'
        ]:
            info[key] = getattr(layer, key)

        return info


def conv_with_kaiming_uniform(norm_cfg: Dict[str, Any] = None,
                              act_cfg: Dict[str, Any] = None,
                              use_deformable: bool = False,
                              use_sep: bool = False) -> Callable:

    def make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) -> nn.Sequential:

        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1

        conv_args = {
            "type": "conv",
            "kernel_size": kernel_size,
            "stride": 1,
            "dilation": 1,
            "padding": dilation * (kernel_size - 1) // 2,
            "groups": groups,
            "bias": (norm_cfg is None)
        }

        conv_layer = get_conv_layer(conv_args)
        conv = conv_layer(in_channels, out_channels)
        if not use_deformable:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(conv.weight, a=1)
            if norm_cfg is None:
                nn.init.constant_(conv.bias, 0)

        module = [
            conv,
        ]
        if norm_cfg is not None:
            norm_layer = get_norm_layer(norm_cfg)
            module.append(norm_layer(out_channels))
        if act_cfg is not None:
            act_layer = get_act_layer(act_cfg)
            module.append(act_layer())
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


convs = {"conv": nn.Conv2d, "dwconv": nn.Conv2d, "deform": ops.DeformConv2d, "conv_trans": nn.ConvTranspose2d}

conv_modules = {'conv': ConvModule, 'dwconv': DWConvModule}


def get_conv_modules(conv_cfg: Dict[str, Any],
                     norm_cfg: Dict[str, Any] = None,
                     act_cfg: Dict[str, Any] = None,
                     **kwargs) -> Callable[[Any], Union[ConvModule, DWConvModule]]:

    assert conv_cfg is not None, "conv_cfg must given."

    name = conv_cfg.get('type', conv_cfg.get('name'))
    conv_module = conv_modules[name.lower()]

    args = {"conv_cfg": conv_cfg, "norm_cfg": norm_cfg, "act_cfg": act_cfg}
    if kwargs is not None:
        kwargs.update(conv_cfg)
        nargs = get_valid_args(conv_module, kwargs)
        args.update(nargs)

    conv_module = partial(conv_module, **args)
    return conv_module


def get_conv_layer(conv_cfg: Dict[str, Any]) -> Callable[[Any], Union[nn.Conv2d, ops.DeformConv2d, nn.ConvTranspose2d]]:
    name = conv_cfg.get('name', conv_cfg.get('type'))
    if name is not None:
        name = name.lower()
    assert name in convs.keys()
    conv_layer = convs[name]

    args = {}
    if conv_cfg is not None:
        nargs = get_valid_args(conv_layer, conv_cfg)
        args.update(nargs)
        conv_layer = partial(conv_layer, **args)

    return conv_layer


class Scale(nn.Module):

    def __init__(self, init_value=1.0, use_scale=True) -> None:
        super().__init__()
        if use_scale:
            self.scale = nn.Parameter(torch.FloatTensor([init_value]))
        else:
            self.register_buffer('scale', torch.FloatTensor([init_value]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        return input * self.scale


class ModuleListDial(nn.ModuleList):

    def __init__(self, modules=None) -> None:
        super().__init__(modules)
        self.cur_position = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0

        return result
