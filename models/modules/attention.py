import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
from modelzipper.tutils import *

from typing import Tuple

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()
    # torch.polar的文档, https://pytorch.org/docs/stable/generated/torch.polar.html
    # torch.polar输入参数是abs和angle，abs所有值都一样，abs和angle的shape都一样
    # torch.polar输入参数是abs和angle，则freqs_cis = abs*(cos(angle) + sin(angle)i)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域,  xq_.shape = [batch_size, seq_len, dim // 2]
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2) #从dim=2维度开始拍平
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, head_dim=1, layer_idx=0, max_len=512):
        super().__init__()
        self.dropout_p = attention_dropout
        self.head_dim = head_dim
        self.layer_idx = layer_idx
 

    def forward(self, qkv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
        """
        qkv = rearrange(
            qkv, "... (three  d) -> ... three  d", three=3, d=self.head_dim
        )
        seqlen = qkv.shape[1]
        
        q, k, v = qkv.unbind(dim=2)
        dim = q.shape[2]
        # import pdb;pdb.set_trace()

        freqs_cis = precompute_freqs_cis(dim=dim, seq_len=seqlen, theta=10000.0).to(q.device).to(q.dtype)
        q, k = apply_rotary_emb(q,k,freqs_cis=freqs_cis)


        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("btd,bsd->bts", q, k * softmax_scale)
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)

        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bts,bsd->btd", attention_drop, v)
        output = rearrange(output, "...  d -> ... ( d)")
        return output

class MHA(nn.Module):
    """Multi-head self-attention
    """

    def __init__(
        self,
        config, 
        layer_idx,
        conv1d_config=None,
    ) -> None:
        super(MHA, self).__init__()
        d_model = config.intermediate_size
        num_heads = conv1d_config.get("att_num_head", 1)
        bias = conv1d_config.get("att_bias", True)
        dropout = conv1d_config.get("att_dropout", 0.1)
        self.window_size = conv1d_config.get("window_size", None)

        assert (
            d_model % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.Wqkv = nn.Linear(
            d_model, 3 * d_model, bias=bias
        )

        if not self.window_size:
            self.inner_attn = SelfAttention(attention_dropout=dropout, head_dim=self.head_dim, layer_idx=layer_idx)
            if layer_idx == 0: log_c("Using Self-Attention")

        self.out_proj = nn.Linear(d_model, d_model)  # Ensure this line is included
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor):
        """Forward pass for Multi-Head Attention"""

        x = x.transpose(1, 2)
        res = x
        qkv = self.Wqkv(x)

        context = self.inner_attn(qkv)
        out = self.out_proj(context)
        out = self.layernorm(out + res)
        out = out.transpose(1, 2)

        return out
