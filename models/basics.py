import torch
import torch.nn as nn
import numpy as np
import math



def conv(batchNorm, in_channels, out_channels, kernel_size=3, stride=1,
        dropout=0.0):
    if batchNorm:
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(dropout)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
                )


def batch_to(dst, *args):
    return [x.to(dst) if x is not None else None for x in args]

def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
