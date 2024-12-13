import torch
import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from transformers import MambaPreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from transformers import PretrainedConfig
from dataclasses import dataclass
from modelzipper.tutils import *

import torch.nn.functional as F
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update)
)


class MetaTransformerConfig(PretrainedConfig):
    model_type = "metatransformer"
    
    def __init__(
        self,
        vocab_size=50280,
        hidden_size=768,
        state_size=16,
        num_hidden_layers=32,
        layer_norm_epsilon=1e-5,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        rescale_prenorm_residual=False,
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)


@dataclass
class MetaOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    cache_params: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MetaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cache_params: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MetaCache:
    def __init__(self, config, batch_size, conv1d_configs = None, dtype=torch.float16, device=None):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        conv_kernel_size = config.conv_kernel
        if conv1d_configs is not None:
            if isinstance(conv1d_configs, dict):
                conv_kernel_size = conv1d_configs['kernel_sizes']
            else:
                conv_kernel_size = conv1d_configs.kernel_sizes
        if isinstance(conv_kernel_size, list):
            self.conv_states = {
                i: [torch.zeros(batch_size, intermediate_size, conv_size, device=device, dtype=dtype)] for conv_size in conv_kernel_size
                for i in range(config.num_hidden_layers)
            }
        else:
            self.conv_states = {
                i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
                for i in range(config.num_hidden_layers)
            }


class MetaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    

def apply_norm(x, norm, batch_norm=False):
    if batch_norm:
        return norm(x)
    else:
        return norm(x.transpose(-1, -2)).transpose(-1, -2)
        

class CustomMultiresLayer(nn.Module):
    def __init__(self, d, num_heads, filter_size, max_seq_len=4096, depth=None, dropout=0.2):
        super(CustomMultiresLayer, self).__init__()
        """
            Args:
            x: input of shape (batch_size, n_channels, sequence_length).
            h0, h1: convolution filters of shape (n_channels, 1, filter_size).
            w: weights of the linear layer after TreeSelect. Shape: (n_channels, depth + 2).
            depth: depth of MultiresConv, i.e., J.
            Returns:
            y: output of shape (batch_size, n_channels, sequence_length).
        """
        self.d = d
        self.filter_size = filter_size
        self.num_heads = num_heads
        if depth is None:
            self.depth = math.ceil(math.log2((max_seq_len - 1) / (filter_size - 1) + 1))
        else:
            self.depth = depth

        self.h0 = nn.Parameter(torch.randn(num_heads, 1, filter_size))  # global info
        self.h1 = nn.Parameter(torch.randn(num_heads, 1, filter_size))  # detail info
        
        self.decay_factor = 1.0
        activation_scaling = 2
        
        self.gate_module = nn.Sequential(
            nn.Linear(d, d // activation_scaling),  # 压缩特征维度
            nn.Sigmoid(),
        )
        
        self.mix_channel_module = nn.Sequential(
            nn.Dropout1d(dropout),
            nn.Conv1d(d, d, 1),
        )
        
        self.norm = nn.LayerNorm(d)

    def _initialize_weights(self):
        #  init self.w with Xavier method
        nn.init.uniform_(self.w, a=-np.sqrt(1. / (self.d * self.depth)), b=np.sqrt(1. / (self.d * self.depth)))

    def forward(self, x):
        dtype = x.dtype
        a = x  # B x d x L
        dilation = 1

        memory = []
        for i in range(self.depth, 0, -1):  # token shift operation
            a = F.pad(a, (dilation * (self.filter_size - 1), 0), "constant", 0)
            b = F.conv1d(a, self.h1, dilation=dilation, groups=self.num_heads)
            a = F.conv1d(a, self.h0, dilation=dilation, groups=self.num_heads)
            memory.append((a, b))
            dilation *= 2
         
        gated_memory, y = self.gated_with_global_info(memory)
        
        channel_mix_y = self.mix_channel_module(y)  # conduct channel mixing
        channel_mix_y = channel_mix_y + x
        channel_mix_y = self.norm(channel_mix_y.transpose(1, 2))
        channel_mix_y = channel_mix_y.to(dtype)
        return channel_mix_y.transpose(1, 2)
        

    def gated_with_global_info(self, memory):
        y = 0
        memory_reversed = memory[::-1]
        gated_memory = []
        for i in range(len(memory_reversed)):
            a, _ = memory_reversed[i]
            if i + 1 < len(memory_reversed):
                _, next_b = memory_reversed[i + 1]  # next local information
                # utilize top (global) state to filter the local infomation
                # non-linear gating
                # token mixing
                gate = torch.sigmoid(a)  
                gated_b = next_b * gate
            gated_memory.append((a, gated_b))
            y += gated_b
        gated_memory = gated_memory[::-1]
        return gated_memory, y
    
    def dynamic_treeselect(self, B, x, memory):
        # utilize global information to filter the local information
        local_info, global_info = memory[-1][0]

        res = 0
        new_selected_memory = []
        local_content, last_context = memory[-1][0], memory[0][0]
        for a, b in memory:
            x, x_gated = torch.split(b, b.size(1) // 2, dim=1)
            x = x * F.sigmoid(x_gated)
            new_selected_memory.append(x)
            res += x
        return res, new_selected_memory


    def custom_treeselect_2(self, memory, L):
        selected = []
        for a, b in memory:
            decay_factors = self.compute_custom_decay_factors(L, b)
            decayed_coefficients = b * decay_factors
            selected.append((a, decayed_coefficients))
        return selected
    

    def compute_custom_decay_factors(self, L, b):
        position_embeddings = torch.arange(L).float() 
        decay_factors = torch.sigmoid(-position_embeddings / (L / 4.0)) 
        decay_factors = decay_factors * (1 + position_embeddings / (L / 10.0))  
        decay_factors = decay_factors.unsqueeze(0)
        decay_factors = decay_factors.repeat(b.size(0), 1)
        return decay_factors


class GatedMultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(GatedMultiScaleConv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad1d((kernel_size - 1, 0), 0),
                nn.Conv1d(
                    in_channels, 
                    out_channels // len(kernel_sizes) * 2, 
                    kernel_size, 
                    groups=out_channels // len(kernel_sizes) * 2,
                ),  # Double the output channels
            ) for kernel_size in kernel_sizes
        ])

    def single_tensor_forward(self, x):
        outputs = []
        for conv in self.convs:
            conv_output = conv(x)  # x: [1, 2048, 4096]
            gate, output = torch.split(conv_output, conv_output.size(1) // 2, dim=1)  # Split the output into two equal parts
            gate = torch.sigmoid(gate)
            outputs.append(output * gate)  # [B, L, D // n]
        outputs = torch.cat(outputs, dim=1)  # concate all the dimension hidden states
        return outputs
    
    def multi_tensor_forward(self, x: List[torch.Tensor]):
        outputs = []
        for conv, hidden_states in zip(self.convs, x):
            conv_output = conv[1](hidden_states)
            gate, output = torch.split(conv_output, conv_output.size(1) // 2, dim=1)
            gate = torch.sigmoid(gate)
            outputs.append(output * gate)
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.single_tensor_forward(x)
        return self.multi_tensor_forward(x)


class vanilla_conv1d_fft(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=4, dilation=1, add_bias=False) -> None:
        super().__init__()
        assert in_channels % groups == 0, "groups must be divided by in_channels"
        self.dilation = dilation
        self.groups = groups
        self.stride = stride
        self.bias = None
        self.weight = torch.empty(out_channels, in_channels // groups, kernel_size)

        # Initialize biases
        if add_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(torch.tensor(fan_in))
            self.bias = nn.Parameter(torch.empty(out_channels).uniform_(-bound, bound))

        nn.init.kaiming_uniform_(self.weight)  # He initialization

    def fft_conv1d_casual(self, x):
        batch_size, in_channels, length = x.shape
        out_channels, group_channels, kernel_size = self.weight.shape

        # 计算需要填充的长度，使得输入序列长度是卷积核大小的倍数
        padding_length = ((kernel_size - 1) - (length - 1)) % kernel_size
        input_padded = torch.nn.functional.pad(x, (padding_length, 0))

        dilated_kernel_size = kernel_size * self.dilation
        weight_dilated = torch.zeros(out_channels, group_channels, dilated_kernel_size, dtype=x.dtype, device=x.device)
        weight_dilated[:, :, (dilated_kernel_size - kernel_size) // 2:(dilated_kernel_size - kernel_size) // 2 + kernel_size] = self.weight

        # devide the input into different groups
        input_reshaped = input_padded.view(batch_size, self.groups, group_channels, -1)

        # FFT
        input_fft = torch.fft.fft(input_reshaped, dim=-1)
        weight_fft = torch.fft.fft(weight_dilated, dim=-1, norm='ortho')
        import pdb; pdb.set_trace()
        output_fft = input_fft * weight_fft.unsqueeze(2)  # Broadcasting over the batch size
        output = torch.fft.ifft(output_fft, dim=-1)

        output_time = output_time.view(batch_size, self.groups, group_channels, -1)
        output = output_time[:, :, :, :length].contiguous()

        if self.bias is not None:
            output += self.bias.view(1, -1, 1)

        # stride
        if self.stride > 1:
            output = output[:, :, ::self.stride]

        output = output.view(batch_size, out_channels, -1)
        import pdb; pdb.set_trace()
        return output  # [B, D, L]
    
    def forward(self, x):
        return self.fft_conv1d_casual(x)


class MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(intermediate_size, embed_dim)
        self.c_proj = nn.Linear(embed_dim, intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MetaMixer(nn.Module):
    def __init__(self, config, layer_idx, conv1d_configs=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel  
        self.intermediate_size = config.intermediate_size
        self.layer_idx = layer_idx
        # only support ``multi_gated_conv1d / multi_vanilla_conv1d / gated_conv1d / vanilla_conv1d'' currently
        self.token_mixer_type = conv1d_configs.token_mixer_type  
        self.multi_gated_conv1d = False
        self.vanilla_conv1d = False

        """
        Module 1. Input-Output Projection
        """
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        
        """     
        Module 2. Token Mixing Module (Maybe save some short-term knowledge here)
        """
        if self.token_mixer_type == "multi_gated_conv1d":
            self.conv1d = CustomMultiresLayer(
                d=config.intermediate_size,
                num_heads=config.intermediate_size,
                filter_size=conv1d_configs['filter_size'],
                depth=conv1d_configs['depth'],
            )
            self.multi_conv1d = True 
        
        elif self.token_mixer_type == "vanilla_conv1d" or "multi_vanilla_conv1d":
            if isinstance(conv1d_configs, dict):
                kernel_sizes = conv1d_configs['kernel_sizes']
            else:
                kernel_sizes = conv1d_configs.kernel_sizes

            if isinstance(kernel_sizes, int):
                self.conv_kernel_size = kernel_sizes
                self.conv1d = nn.Sequential(
                    nn.ConstantPad1d((kernel_sizes - 1, 0), 0),
                    nn.Conv1d(
                        in_channels=self.intermediate_size,
                        out_channels=self.intermediate_size,
                        kernel_size=kernel_sizes,
                        groups=8,  # default for 8 heads
                        stride=1,
                        dilation=1,
                        bias=config.use_conv_bias,
                    )
                )
                
                # self.conv1d = vanilla_conv1d_fft(
                #     in_channels=self.intermediate_size,
                #     out_channels=self.intermediate_size,
                #     kernel_size=kernel_sizes,
                #     groups=self.intermediate_size,
                #     stride=kernel_sizes - 1,
                #     dilation=1,
                #     add_bias=config.use_conv_bias,
                # )

            elif isinstance(kernel_sizes, list):  
                self.conv_kernel_size = kernel_sizes
                self.conv1d = GatedMultiScaleConv1d(
                    config.intermediate_size, 
                    config.intermediate_size, 
                    kernel_sizes
                )
                self.multi_conv1d = True  # use multi_conv1d_forward
            else:
                raise ValueError("Invalid kernel_sizes (<=4) for GatedMultiScaleConv1d or utilize custom module")
        
        else:
            raise NotImplementedError
        
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.ln_2 = nn.LayerNorm(self.intermediate_size, eps=config.layer_norm_epsilon)

        """
        Module 3. Channel Mixing Module (Maybe save some long-term knowledge here)
        """
        self.mlp = MLP(self.intermediate_size, config)



    def norm_forward(self, hidden_states, cache_params=None, extra_kwargs=None):
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Gated MLP's linear projection (extra serves as gated)
        projected_states = self.in_proj(hidden_states).transpose(1, 2) # [B, 2D, L]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation (token mixing)
        if cache_params is not None:
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx] # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx] = conv_state.clone()
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)

                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1) # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(  # only save last conv_kernel_size states
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state.clone()
                hidden_states = self.conv1d(hidden_states)[..., :seq_len] # [batch, intermediate_size, seq_len]
        else:
            hidden_states = self.conv1d(hidden_states)[..., :seq_len] # [B, D, L]

        # 3. Gated Convolution (attention / filtering)
        hidden_states = hidden_states * self.act(gate) # [B, D, L]

        # 4. MLP forward (channel mixing)
        hidden_states = hidden_states.transpose(1, 2) # [B, L, D]
        residule = hidden_states 
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states) + residule  # residual connection 

        # 5. Final linear projection
        contextualized_states = self.out_proj(hidden_states) # [B, L, D]
        
        return contextualized_states


    def forward(self, hidden_states, cache_params=None, extra_kwargs=None):
        return self.norm_forward(
            hidden_states, 
            cache_params, 
            extra_kwargs=extra_kwargs['extra_kwargs'] if 'extra_kwargs' in extra_kwargs else None
        )
        

class MetaBlock(nn.Module):
    def __init__(self, config, layer_idx, conv1d_configs=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MetaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MetaMixer(config, layer_idx, conv1d_configs)

    def forward(self, hidden_states, cache_params=None, extra_kwargs=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, cache_params=cache_params, extra_kwargs=extra_kwargs)
        hidden_states = residual + hidden_states
        return hidden_states


class MetaModel(MambaPreTrainedModel):
    def __init__(self, config, conv1d_configs=None) -> None:
        super().__init__(config)
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.max_position_embeddings = max_position_embeddings
        # self.use_relative_position = use_relative_position
        # self.use_abs_position = use_abs_position
        
        
        self.layers = nn.ModuleList(
            [
                MetaBlock(
                    config, 
                    layer_idx=idx,
                    conv1d_configs=conv1d_configs,
                ) for idx in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False
        self.norm_f = MetaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MetaOutput]:
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
   
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MetaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        position_embeds = None
        # input_shape = input_ids.shape
        # if self.use_relative_position or self.use_abs_position:
        #     if position_ids is None:
        #         if cache_params is not None:
        #             position_ids = torch.arange(cache_params.seqlen_offset, input_shape[-1] + cache_params.seqlen_offset, dtype=torch.long).to(input_ids.device)
        #             position_ids = position_ids.unsqueeze(0)
        #         else:
        #             position_ids = torch.arange(input_shape[-1], dtype=torch.long).to(input_ids.device)
        #             position_ids = position_ids.unsqueeze(0)

        #     if self.use_abs_position:
        #         position_embeds = self.wpe(position_ids).to(inputs_embeds.dtype)
        #     elif self.use_relative_position:  # TODO: DEBUG
        #         freqs = self.freqs.float().to(position_ids.device)
        #         position_ids = position_ids.float()
        #         # Add a small constant to avoid division by zero
        #         freqs += 1e-7
        #         angles = position_ids.unsqueeze(-1) / freqs.unsqueeze(0)
        #         position_embeds = torch.cat([angles.sin(), angles.cos()], dim=-1).to(inputs_embeds.dtype)

        hidden_states = inputs_embeds + position_embeds if position_embeds is not None else inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params, extra_kwargs=extra_kwargs)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params, extra_kwargs=extra_kwargs)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)
        # import pdb;pdb.set_trace()
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MetaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class MetaTransformerForCausalLM(MambaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, custom_conv1d_configs=None) -> None:
        super().__init__(config)
        self.backbone = MetaModel(config, conv1d_configs=custom_conv1d_configs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        
    def custom_from_pretrained(self, path, dtype, is_from_pytorch_lightning=False, strict=False):
        if self.dtype != dtype:
            self.to(dtype)
        state_dict = torch.load(path, map_location='cpu')
        if state_dict.get('state_dict'):
            state_dict = state_dict['state_dict']
        if dtype is not None:
            state_dict = {k: v.type(dtype) for k, v in state_dict.items()}
        if is_from_pytorch_lightning:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('model.', '')] = v
            state_dict = new_state_dict
        self.load_state_dict(state_dict, strict=strict)
        
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs["cache_params"]
        return model_kwargs
    
    def _retrieve_from_memory(self, query_states, dense_mem=None):
        if dense_mem is None:
            return torch.zeros_like(query_states)
        query_states = F.elu(query_states) + 1  # ELU activation
        memory_output = torch.matmul(query_states, dense_mem) / self.norm_term
        return memory_output


    def _update_memory(self, key_states, value_states):
        key_states = F.elu(key_states) + 1  # ELU activation
        if self.memory is not None:
            self.memory = self.memory + torch.matmul(
                key_states.transpose(-2, -1), value_states
            )
        else:
            self.memory = torch.matmul(key_states.transpose(-2, -1), value_states)
        if self.norm_term is not None:
            self.norm_term = self.norm_term + torch.unsqueeze(key_states.sum(dim=-2),-2)
        else:
            self.norm_term = torch.unsqueeze(key_states.sum(dim=-2),-2)


    def prepare_inputs_for_generation(
        self, input_ids, cache_params=None, inputs_embeds=None, attention_mask=None, extra_kwargs=None, **kwargs,
    ):  
        """
        extra_kwargs: for analysis like depth and ctx_length
        """
        model_inputs = {"input_ids": None, "inputs_embeds": None}

        if extra_kwargs['mode'] == 'chunk-wise':
            assert extra_kwargs['segment_size'] is not None, "must define segment size"
            if not hasattr(self, "memory"): self.memory = {}
            if not hasattr(self, "norm_term"): self.norm_term = {}
            self.segment_size = extra_kwargs['segment_size']
            total_len = input_ids.size(-1)
            dense_mem = torch.tensor_split(
                input_ids,
                list(range(self.segment_size, total_len, self.segment_size)),
                dim=1,
            )
            model_inputs["dense_mem"] = dense_mem

        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache_params"] = cache_params
        model_inputs["extra_kwargs"] = extra_kwargs
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_kwargs=kwargs,  # for analysis like depth and ctx_length
        )
        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MetaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )


class conv1d_configs:
    kernel_sizes = 4
    token_mixer_type = "vanilla_conv1d"


if __name__ == "__main__":
    model_path = "/UNICOMFS/hitsz_khchen_4/zecheng/hf_models/mamba-370m-hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    config = transformers.MambaConfig.from_pretrained(model_path)

    model = MetaTransformerForCausalLM(config, custom_conv1d_configs=conv1d_configs())
    
    # raw_model = transformers.MambaForCausalLM.from_pretrained("/nvme/hf_models/mamba-370m-hf")
    # state_dict = raw_model.state_dict()
    # model.load_state_dict(state_dict, strict=False)
    # model.custom_from_pretrained(
    #     "/nvme/hf_models/mamba-370m-hf/pytorch_model.bin",
    #     dtype=torch.bfloat16,
    #     is_from_pytorch_lightning=True,
    # )
    
    model = model.train().cuda()    

    # data = auto_read_data("/UNICOMFS/hitsz_khchen_4/zecheng/data/longbench/data/qasper.jsonl")
    # print_c(data[0]['input'])

    # input_ids = tokenizer(data[0]['context'] + " " + data[0]['input'], return_tensors="pt").input_ids.cuda()

    input_ids = torch.randint(0, 20000, (1, 1024)).long().cuda()
    
    res = model(
        input_ids=input_ids,
        labels=input_ids,
        # max_length=input_ids.size(-1) + 64,
        # min_length=input_ids.size(-1) + 16,
    )
    
    print(res)
    print_c(res.loss)
    
    