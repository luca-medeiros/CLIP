from collections import OrderedDict

import torch
from torch import nn


class LayerNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.inner_layernorm = nn.LayerNorm(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = self.inner_layernorm(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        length = x.shape[0]
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask[:length, :length])[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, low_dim: int, num_tokens: int, context_length: int = 30):
        super().__init__()
        self.width = width
        self.layers = layers
        self.context_length = context_length
        res_block = [ResidualAttentionBlock(self.width, heads, self.build_attention_mask()) for _ in range(self.layers)]
        self.resblocks = nn.Sequential(*res_block)
        self.token_embedding = nn.Embedding(num_tokens, self.width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, self.width))
        self.ln_final = LayerNorm(self.width)
        self.text_projection = nn.Parameter(torch.empty(self.width, low_dim))
        self.initialize_parameters()

    def forward(self, text):
        max_val = text.argmax(dim=-1)
        x = self.token_embedding(text)

        x = x + self.positional_embedding[:x.shape[1]]

        x = x.permute(1, 0, 2)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), max_val] @ self.text_projection
        return x

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.width**-0.5) * ((2 * self.layers)**-0.5)
        attn_std = self.width**-0.5
        fc_std = (2 * self.width)**-0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


def clip_transformer_base(num_tokens, **kwargs):
    return Transformer(512, 12, 8, 512, num_tokens)


def clip_transformer_large(num_tokens, **kwargs):
    return Transformer(768, 12, 12, 768, num_tokens)
