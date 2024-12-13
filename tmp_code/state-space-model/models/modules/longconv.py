import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
from modelzipper.tutils import * 
from models.modules.mamba_analysis_utils import *

import torch

class Longconv(nn.Module):
    def __init__(self, config, layer_idx, conv1d_config=None):
        super().__init__()
        self.conv1d_config = conv1d_config

        self.long_kernel_size  = conv1d_config.get("long_conv_kernel", 128)
        if not self.long_kernel_size : self.long_kernel_size  = 128
        
        self.module =  nn.Conv1d(in_channels=config.intermediate_size,
            out_channels = config.intermediate_size,
            bias = config.use_conv_bias,
            kernel_size = self.long_kernel_size ,
            groups = config.intermediate_size,
            padding= self.long_kernel_size -1)
        # self.lambda_squash

        if layer_idx == 0: log_c(f"Using Longconv-{self.long_kernel_size }")
        self.layer_idx = layer_idx

    def forward(self, x, cache_params=None):
        seq_len = x.shape[-1]
        # if cache_params is not None:
        #     # import pdb;pdb.set_trace()
        #     if cache_params.seqlen_offset > 0:
        #         # import pdb;pdb.set_trace()
        #         long_conv_state = cache_params.long_conv_state[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
        #         long_conv_state = torch.roll(long_conv_state, shifts=-1, dims=-1)
        #         long_conv_state[:, :, -1] = x[:, :, 0]
        #         cache_params.long_conv_state[self.layer_idx].copy_(long_conv_state)
        #         output = torch.sum(long_conv_state * self.module.weight[:, 0, :], dim=-1)
        #         # x += self.module.bias
        #         output = output.to(x.dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
        #     else:
        #         long_conv_state = nn.functional.pad(
        #             x,
        #             (self.long_kernel_size - x.shape[-1], 0)
        #         )
        #         cache_params.long_conv_state[self.layer_idx].copy_(long_conv_state)
        #         output = self.module(x)[..., :seq_len]     # [batch, intermediate_size, seq_len]
        # else:
        output = self.module(x)[..., :seq_len]        # [batch, intermediate_size, seq_len]

        return output
    
