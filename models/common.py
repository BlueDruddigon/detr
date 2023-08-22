from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: Optional[int] = None,
            act_layer: nn.Module = nn.ReLU6,
            drop_prob: float = 0.
    ) -> None:
        out_features = out_features or in_features
        layers = nn.ModuleList([
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop_prob),
        ])
        super(MLP, self).__init__(*layers)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = self.embed_dim // self.num_heads
        self.scale = head_dim ** -0.5

        # Create the query, key, value projection layers
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # Create the output projection layer
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Compute the query, key and value projections
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Scale Dot-Product Attention
        attn_output = F.scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=mask, dropout_p=0.0)

        return self.proj_drop(self.proj(attn_output))
