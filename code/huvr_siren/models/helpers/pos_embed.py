# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# MAE: https://github.com/facebookresearch/mae
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import numpy as np
import torch


def get_2d_sincos_pos_embed_rectangle(embed_dim, grid_size,cls_token=False,num_registers=0):
    """
    grid_size, a tuple of height and width
    """
    grid_size_h, grid_size_w = grid_size
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_w, grid_size_h])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1+num_registers, embed_dim]), pos_embed], axis=0)
    elif num_registers:
        pos_embed = np.concatenate([np.zeros([num_registers, embed_dim]), pos_embed],axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed_direct(pos_embed, orig_size, new_size, mode="bicubic"):
    #pos_embed: [B, C, H, W]
    #orig_size: int of the original size
    #new_size: int of the new size
    #return: [B, C, H, W]
    orig_h, orig_w = orig_size
    new_h, new_w = new_size
    _, seq_len, embedding_size = pos_embed.shape
    assert seq_len == orig_h * orig_w

    pos_embed = pos_embed.reshape(-1, orig_h, orig_w, embedding_size).permute(0, 3, 1, 2)
    #change to [B, C, H, W], C is embedding_size
    pos_embed = torch.nn.functional.interpolate(
        pos_embed, size=(new_h, new_w), mode=mode, align_corners=False)
    pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
    return pos_embed


