import functools
import inspect
import torch
import torch.nn as nn
import numpy as np

from typing import Any, Callable, Dict

def get_valid_args(cls_or_func: Callable, cfg: Dict) -> Dict[str, Any]:
    valid_args = {
        key: value for key, value in cfg.items() \
            if key in inspect.signature(cls_or_func).parameters.keys()}
    return valid_args


def init_weight_only(func):

    @functools.wraps(func)
    def wrapper(module, **kwargs):
        if isinstance(module, nn.Identity):
            return

        if hasattr(module, 'weight'):
            func(module.weight, **kwargs)

            if 'bias' in kwargs and hasattr(module, 'bias'):
                nn.init.constant_(module.bias, kwargs.get('bias'))
        else:
            func(module, **kwargs)

    return wrapper


def bias_init_with_prob(prior_prob):
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


@init_weight_only
def constant_init(weight, val, **kwargs):
    nn.init.constant_(weight, val)


@init_weight_only
def normal_init(weight, mean=0, std=1, **kwargs):
    nn.init.normal_(weight, mean, std)


@init_weight_only
def xavier_uniform_init(weight, gain=1.0, **kwargs):
    nn.init.xavier_uniform_(weight, gain)


@init_weight_only
def kaiming_init(weight, a=0, mode='fan_out', nonlinearity='relu', distribution='normal', **kwargs):
    assert distribution in ['uniform', 'normal']

    if distribution == 'uniform':
        nn.init.kaiming_uniform_(weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(weight, a=a, mode=mode, nonlinearity=nonlinearity)


@init_weight_only
def trunc_normal_init(tensor, mean=0., std=1., **kwargs):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class Hook:

    def __init__(self, module, backward=False):
        if not backward:
            self.hook = module.register_forward_hook(self.forward_hook)
        else:
            self.hook = module.register_backward_hook(self.backward_hook)

    @staticmethod
    def validate(data):
        if torch.isinf(data).sum() > 0:
            return False
        if torch.isnan(data).sum() > 0:
            return False
        return True

    def forward_hook(self, m, input, output):
        self.input = input
        self.output = output

    def backward_hook(self, m, grad_in, grad_out):
        self.grad_in = grad_in
        self.grad_out = grad_out

    def close(self):
        self.hook.remove()
