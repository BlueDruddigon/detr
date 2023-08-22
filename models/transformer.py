from typing import Optional

import torch
import torch.nn as nn
from torchvision.ops.stochastic_depth import StochasticDepth

from .common import MLP, MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      num_hidden: int,
      qkv_bias: bool = False,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      drop_path: float = 0.1,
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(
          embed_dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        self.ffn = MLP(embed_dim, num_hidden)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        self.drop_path = StochasticDepth(p=drop_path, mode='row')
    
    @staticmethod
    def __with_pos_embed(x: torch.Tensor, pos: Optional[torch.Tensor] = None):
        return x if pos is None else x + pos
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None):
        q = k = self.__with_pos_embed(src, pos)
        # Multi-Head Self-Attention
        attn = self.attn(query=q, key=k, value=src, mask=mask)
        attn = self.attn_norm(src + self.drop_path(attn))
        # FFN
        out = self.ffn(attn)
        out = self.ffn_norm(attn + self.drop_path(out))
        return out


class DecoderLayer(nn.Module):
    def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      num_hidden: int,
      qkv_bias: bool = False,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      drop_path: float = 0.1
    ) -> None:
        super(DecoderLayer, self).__init__()
        
        # Masked Multi-Head Self-Attention
        self.masked_attn = MultiHeadAttention(
          embed_dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.masked_attn_norm = nn.LayerNorm(embed_dim)
        
        # Cross Multi-Head Attention
        self.cross_attn = MultiHeadAttention(
          embed_dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = MLP(embed_dim, num_hidden)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        # Stochastic depth
        self.drop_path = StochasticDepth(p=drop_path, mode='row')
    
    @staticmethod
    def __with_pos_embed(x: torch.Tensor, pos: Optional[torch.Tensor] = None):
        return x if pos is None else x + pos
    
    def forward(
      self,
      tgt: torch.Tensor,
      memory: torch.Tensor,
      tgt_mask: Optional[torch.Tensor] = None,
      memory_mask: Optional[torch.Tensor] = None,
      query_pos: Optional[torch.Tensor] = None,
      pos: Optional[torch.Tensor] = None
    ):
        q = k = self.__with_pos_embed(tgt, query_pos)
        # Masked Multi-Head Self-Attention
        attn = self.masked_attn(query=q, key=k, value=tgt, mask=tgt_mask)
        attn = self.masked_attn_norm(tgt + self.drop_path(attn))
        # Cross Multi-Head Attention
        shortcut = attn
        q = self.__with_pos_embed(attn, query_pos)
        k = self.__with_pos_embed(memory, pos)
        attn = self.cross_attn(query=q, key=k, value=memory, mask=memory_mask)
        attn = self.cross_attn_norm(shortcut + self.drop_path(attn))
        # FFN
        out = self.ffn(attn)
        out = self.ffn_norm(attn + self.drop_path(out))
        return out


class Encoder(nn.Module):
    def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      num_hidden: int,
      num_layers: int,
      qkv_bias: bool = False,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      drop_path: float = 0.1,
      pre_norm: bool = False
    ) -> None:
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
          EncoderLayer(
            embed_dim,
            num_heads,
            num_hidden,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path
          ) for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None):
        out = src
        
        for layer in self.layers:
            out = layer(out, mask=mask, pos=pos)
        
        out = self.norm(out)
        return out


class Decoder(nn.Module):
    def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      num_hidden: int,
      num_layers: int,
      qkv_bias: bool = False,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      drop_path: float = 0.1,
      pre_norm: bool = False,
      return_intermediate: bool = False
    ) -> None:
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
          DecoderLayer(
            embed_dim,
            num_heads,
            num_hidden,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path
          ) for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
        self.return_intermediate = return_intermediate
    
    def forward(
      self,
      tgt: torch.Tensor,
      memory: torch.Tensor,
      tgt_mask: Optional[torch.Tensor] = None,
      memory_mask: Optional[torch.Tensor] = None,
      query_pos: Optional[torch.Tensor] = None,
      pos: Optional[torch.Tensor] = None
    ):
        out = tgt
        intermediate = []
        
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, query_pos=query_pos, pos=pos)
            if self.return_intermediate:
                intermediate.append(self.norm(out))
        
        out = self.norm(out)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(out)
            return torch.stack(intermediate)
        
        return out.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(
      self,
      embed_dim: int = 512,
      num_heads: int = 8,
      num_encoder_layers: int = 6,
      num_decoder_layers: int = 6,
      num_hidden_features: int = 2048,
      qkv_bias: bool = False,
      attn_drop: float = 0.1,
      proj_drop: float = 0.1,
      drop_path: float = 0.1,
      pre_norm: bool = False,
      return_intermediate: bool = False
    ) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(
          embed_dim,
          num_heads,
          num_hidden_features,
          num_encoder_layers,
          qkv_bias=qkv_bias,
          attn_drop=attn_drop,
          proj_drop=proj_drop,
          drop_path=drop_path,
          pre_norm=pre_norm
        )
        
        self.decoder = Decoder(
          embed_dim,
          num_heads,
          num_hidden_features,
          num_decoder_layers,
          qkv_bias=qkv_bias,
          attn_drop=attn_drop,
          proj_drop=proj_drop,
          drop_path=drop_path,
          pre_norm=pre_norm,
          return_intermediate=return_intermediate
        )
        
        self._reset_parameters()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: torch.Tensor, mask: torch.Tensor, query_embed: torch.Tensor, pos_embed: torch.Tensor):
        # N: batch_size, E: embed_dim, S: source sequence length, T: target sequence length
        B, C, H, W = src.shape
        src = src.flatten(2).permute(0, 2, 1)  # N, S, E
        pos_embed = pos_embed.flatten(2)  # N, S, E
        query_embed = query_embed.unsqueeze(1).permute(1, 0, 2)  # N, T, E
        mask = mask.flatten(1)  # N, S
        
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs, memory.permute(0, 2, 1).view(B, C, H, W)


def build_transformer(args):
    return Transformer(
      embed_dim=args.embed_dim,
      num_heads=args.num_heads,
      num_encoder_layers=args.enc_layers,
      num_decoder_layers=args.dec_layers,
      num_hidden_features=args.num_hidden_features,
      qkv_bias=args.qkv_bias,
      attn_drop=args.dropout,
      proj_drop=args.dropout,
      drop_path=args.dropout,
      pre_norm=args.pre_norm,
      return_intermediate=True
    )
