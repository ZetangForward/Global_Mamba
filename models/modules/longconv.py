import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
from modelzipper.tutils import * 
from models.modules.mamba_analysis_utils import *

import torch

def position_encoding(tensor, kernel_size):
    """
    使用与序列相关的值生成位置编码。位置0的编码值为序列均值的两倍，
    并在kernel_size处衰减到均值，之后的位置保持为均值。
    
    参数:
    tensor (Tensor): 输入张量，形状为 [B, Seq_len, dimension]
    kernel_size (int): 控制编码衰减的范围
    
    返回:
    Tensor: 添加位置编码后的张量
    """
    B, Seq_len, dimension = tensor.shape
    
    # 生成位置索引
    positions = torch.arange(Seq_len, device=tensor.device).to(tensor.dtype)  # 形状 [Seq_len]
    
    # 计算每个batch的序列均值
    mean_value = tensor.mean(dim=(1, 2), keepdim=True)  # 形状 [B, 1, 1]
    # import pdb;pdb.set_trace()
    # 计算batch相关的 start_value
    start_value = abs(mean_value / tensor[:, 0, :].mean(dim=1, keepdim=True).unsqueeze(-1)) # 形状 [B, 1, 1]
    
    # 计算batch相关的 decay_rate
    decay_rate = torch.log(start_value) / kernel_size  # 形状 [B, 1, 1]
    
    # 计算位置编码
    position_encoding = start_value * torch.exp(-decay_rate * positions.unsqueeze(0).unsqueeze(-1))  # 形状 [B, Seq_len, 1]
    
    position_encoding[:,kernel_size:,] = 1
    # 扩展到所有特征维度
    position_encoding = position_encoding.expand(B, Seq_len, dimension)
    
    # 应用位置编码
    encoded_tensor = tensor * position_encoding
    # import pdb;pdb.set_trace()
    
    return encoded_tensor.transpose(1,2)


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


    def forward(self, x):
        seq_len = x.shape[-1]
        # import pdb;pdb.set_trace()
        output = self.module(x)
        
        # output = position_encoding(output.transpose(1,2), self.long_kernel_size)

        return output[...,:seq_len]
    




        # def replicate_front_padding_batch(self, input_tensor, kernelsize):
    #     batch_size, seq_len = input_tensor.size(0), input_tensor.size(-1)
    #     padding_size = kernelsize - 1
        
    #     # 创建填充部分：从每个序列中提取去掉第一个 token 后的前三个 token
    #     front_padding = input_tensor[:,:, 1:1 + padding_size]
        
    #     # 将提取的部分拼接到每个序列的前面
    #     padded_tensor = torch.cat([front_padding, input_tensor], dim=-1)
        
    #     return padded_tensor
    
    # # def post_process_output(self, output):
    # #     seq_len = x.shape[-1]
    # #     x = torch.sign(x) * torch.clamp(torch.abs(x) - self.lambda_squash, min=0)
    # #             # import pdb;pdb.set_trace()
        # padded_tensor = self.replicate_front_padding_batch(x,self.long_kernel_size)
        # output = self.module(padded_tensor)[:,:, self.long_kernel_size-1: self.long_kernel_size + seq_len -1] 