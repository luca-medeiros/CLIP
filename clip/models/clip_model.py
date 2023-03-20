import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import model_zoo

from clip.models.transformer import clip_transformer_base
from clip.models.vision_transformer import vit_b_p16_224x224


class CLIPModel(nn.Module):

    def __init__(self, num_tokens, low_dim: int = 512, logit_scale: float = 0.07, *args, **kwargs) -> None:
        super().__init__()
        self.img_encoder = vit_b_p16_224x224()
        self.img_encoder.heads = nn.Linear(self.img_encoder.heads[0].in_features, low_dim, bias=False)
        self.txt_encoder = clip_transformer_base(num_tokens)
        self.low_dim = low_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale))

    def encode_img(self, imgs: torch.Tensor) -> torch.Tensor:
        img_feats = self.img_encoder(imgs)
        img_feats = F.normalize(img_feats, dim=-1)
        return img_feats

    def encode_txt(self, txts: torch.Tensor) -> torch.Tensor:
        txt_feats = self.txt_encoder(txts)
        txt_feats = F.normalize(txt_feats, dim=-1)
        return txt_feats

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:

        img_feats = self.encode_img(images)
        txt_feats = self.encode_txt(text)
        logits_per_image = self.logit_scale.exp() * img_feats @ txt_feats.t()

        return logits_per_image


if __name__ == '__main__':
    img = torch.zeros((3, 3, 224, 224))
    text = torch.zeros(3, 10).long()

    clip = CLIPModel(num_tokens=1000)
    print(clip(img, text).shape)
