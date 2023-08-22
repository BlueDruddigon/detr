import math

import torch
import torch.nn as nn


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length: int = 1024) -> None:
        super(AbsolutePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        self.register_buffer('pe', self._create_pe_matrix())
    
    def _create_pe_matrix(self):
        # initialize an empty tensor to store the positional encoding
        pe = torch.zeros(self.max_length, self.embed_dim)
        
        # create a tensor containing the patch indices, shape (max_length, embed_dim / 2)
        position = torch.arange(self.max_length).unsqueeze(1)
        # a^b = e^(b*ln(a))
        # a = 10000, b = i/d_model, ln(a) = math.log(10000)
        div_term = torch.exp(torch.arange(0, self.embed_dim, step=2) * -(math.log(10000.0) / self.embed_dim))
        
        # sinusoidal functions for even and odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (tensor): input tensor of shape (batch_size, num_tokens, embed_dim)

        Returns:
            tensor: positional encoding, shape (batch_size, num_tokens, embed_dim)
        """
        batch_size, num_tokens, _ = x.size()
        # slice the positional encoding matrix according to the num_tokens
        pe = self.pe[:num_tokens]
        # un-squeeze the batch dimension and repeat the positional encoding matrix for each batch element
        return pe.unsqueeze(0).repeat(batch_size, 1, 1)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length: int = 1024) -> None:
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        self.pe = nn.Parameter(torch.randn(self.max_length, self.embed_dim))
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (tensor): input tensor of shape (batch_size, num_tokens, embed_dim)

        Returns:
            tensor: positional encoding, shape (batch_size, num_tokens, embed_dim)
        """
        batch_size, num_tokens, _ = x.size()
        # slice the positional encoding matrix according to the num_tokens
        pe = self.pe[:num_tokens]
        # un-squeeze the batch dimension and repeat the positional encoding matrix for each batch element
        return pe.unsqueeze(0).repeat(batch_size, 1, 1)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length: int = 1024) -> None:
        super(RelativePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        self.pe = nn.Parameter(torch.randn(self.max_length, self.embed_dim))
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (tensor): input tensor of shape (batch_size, num_tokens, embed_dim)

        Returns:
            tensor: positional encoding, shape (batch_size, num_tokens, num_tokens, embed_dim)
        """
        batch_size, num_tokens, _ = x.size()
        # compute the relative distances between the positions
        pos = torch.arange(num_tokens, device=x.device)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        
        # index into the relative positional encoding
        rel_pos = rel_pos + self.max_length // 2
        rel_pos = torch.clamp(rel_pos, 0, self.max_length - 1)
        
        return self.pe[rel_pos].unsqueeze(0).repeat(batch_size, 1, 1, 1)


def build_positional_encoding(args):
    embed_dim = args.embed_dim
    if args.position_embedding in ('v2', 'absolute'):
        position_embedding = AbsolutePositionalEncoding(embed_dim)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = LearnablePositionalEncoding(embed_dim)
    elif args.position_embedding in ('v4', 'relative'):
        position_embedding = RelativePositionalEncoding(embed_dim)
    else:
        raise ValueError(f'{args.position_embedding} is not supported')
    
    return position_embedding
