from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

from utils.misc import NestedTensor
from .pos_embeddings import build_positional_encoding


class Joiner(nn.Module):
    def __init__(
            self,
            name: str,
            pos_embed: nn.Module,
            weights: str = 'DEFAULT',
            train: bool = False,
            dilation: bool = False
    ) -> None:
        super(Joiner, self).__init__()
        backbone = getattr(torchvision.models, name)(
            weights=weights, norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, dilation]
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.num_channels = 512 if name in {'resnet18', 'resnet34'} else 2048
        self.pos_embed = pos_embed

        if not train:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        out = self.backbone(tensor_list.tensor)
        xs: Dict[str, NestedTensor] = {}
        # Feature Capturing
        for idx, x in enumerate([out]):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            xs[str(idx)] = NestedTensor(x, mask)

        out: List[NestedTensor] = []
        pos = []
        for x in xs.values():
            out.append(x)
            pos_tensor = x.tensor.flatten(2).permute(0, 2, 1)
            # positional encoding
            pos.append(self.pos_embed(pos_tensor).to(x.tensor.dtype))
        return out, pos


def build_backbone(args):
    pos_embed = build_positional_encoding(args)
    # bb = Backbone(args.backbone, False, False, args.dilation)
    # j = Joiner2(bb, pos_embed)
    # j.num_channels = bb.num_channels
    # return j
    return Joiner(args.backbone, pos_embed)
