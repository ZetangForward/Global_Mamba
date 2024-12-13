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
from transformers.modeling_utils import PreTrainedModel
from .configuration_mamba import MambaConfig
# from m
# from transformers.cache_utils import MambaCache
from dataclasses import dataclass
from modelzipper.tutils import *
import torch.nn.functional as F
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from models.custom_selective_scan_interface import selective_scan_fn
    from models.custom_selective_scan_interface import mamba_inner_fn
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

from einops import rearrange, repeat


# from models.mamba_analysis_utils import *

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update)
)

def reverse_exponential_decay_fn(i, total_layer, k=1.0):
    decay_value = 1 - math.exp(-k * i / total_layer)
    return decay_value

def exponential_decay_fn(i, total_layer, k=1.0):
    decay_value = math.exp(-k * i / total_layer)
    return decay_value

def decay_fn(i, total_layer):
    decay_value = 1 - (i / (total_layer))
    return decay_value

def apply_norm(x, norm, batch_norm=False):
    if batch_norm:
        return norm(x)
    else:
        return norm(x.transpose(-1, -2)).transpose(-1, -2)


@dataclass
class MambaOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    cache_params: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    orthogonal_losses: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MambaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    total_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cache_params: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    orthogonal_losses: Dict[str, torch.FloatTensor] = None
    total_orthogonal_loss: Optional[Tuple[torch.FloatTensor]] = None


class MambaCache:
    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
        

class MambaMixer(nn.Module):
    def __init__(self, config, layer_idx, conv1d_configs=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel  
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = config.time_step_rank
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )
        
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden statesa
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        self.module = None
        self.decay = None
        self.gated_conv1d = None
        self.directdecay = None
        if conv1d_configs.get("module_type"):
            modules = conv1d_configs['module_type'].split("-")
            for module in modules:
                if module == "longconv":
                    long_kernel_size = conv1d_configs.get("long_conv_kernel", 128)
                    if layer_idx==0: log_c(f"Longconv-{long_kernel_size}")
                    self.module =  nn.Conv1d(in_channels=self.intermediate_size,
                        out_channels = self.intermediate_size,
                        bias = config.use_conv_bias,
                        kernel_size = long_kernel_size,
                        groups = self.intermediate_size,
                        padding= long_kernel_size-1,)

                elif module == "decay":
                    decay_rate = conv1d_configs.get("decay_rate", 0.01)
                    self.decay = decay_rate
                    if layer_idx==0: log_c(f"Decay-{str(decay_rate)}")
                
                elif module == "directdecay":
                    decay_rate = conv1d_configs.get("decay_rate", 0.01)
                    self.directdecay = decay_rate
                    if layer_idx==0: log_c(f"Directdecay-{str(decay_rate)}")

                elif module == "gatedconv":
                    self.gated_conv1d = nn.Conv1d(
                        in_channels=self.intermediate_size,
                        out_channels = self.intermediate_size,
                        bias = config.use_conv_bias,
                        kernel_size = config.conv_kernel,
                        groups = self.intermediate_size,
                        padding= config.conv_kernel-1,)
                    if layer_idx==0:log_c(f"Gatedconv")
                
                else:
                    if layer_idx==0: log_c(f"Wrong Module {module}", colored='red')

        else:
            log_c("No module used")
    

        self.mix_type=conv1d_configs.get("module_type", None)

    def mix_fn(self, dt, n):
        if not self.mix_type:
            return dt * self.act(n)
        
    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params=None, extra_kwargs=None):
        
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        if self.training and cache_params is None:  # Doesn't support outputting the states -> used for training
            # import pdb;pdb.set_trace()
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:  # inference mode
             
            hidden_states, gate = projected_states.chunk(2, dim=1)
            
            # 2. Convolution sequence transformation
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states[self.layer_idx],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)
 
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_states)
                
                hidden_states = causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )


            # 3. State Space Model sequence transformation
            # 3.a. input varying initialization of time_step, B and C
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

            A = -torch.exp(self.A_log.float())
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
            if cache_params is not None and cache_params.seqlen_offset > 0:
                cur_ssm_state = cache_params.ssm_states[self.layer_idx]
                scan_outputs = selective_state_update(
                    cur_ssm_state,
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))

       
        return contextualized_states

    def slow_forward(self, input_states, cache_params=None, extra_kwargs=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype


        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2) # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)          # [batch, intermediate_size, seq_len]


 ############# Custom module sequence transformation   #############
        def create_distant_decay_matrix(seq_len, decay_rate=0.001):
            t = torch.arange(seq_len, dtype=torch.bfloat16)
            decay_weights = torch.exp(-decay_rate * t)
            decay_weights = torch.clamp(decay_weights, min=1e-6)
            return decay_weights.view(1, 1, -1)
        
        if self.decay:
            oscillating_decay_matrix = create_distant_decay_matrix(seq_len).to(hidden_states.device) 
            decay_input =   hidden_states.clone().detach() * oscillating_decay_matrix

        module_output = None
        if self.module:
            module_input = decay_input if self.decay else hidden_states.clone().detach()
            module_input = module_input.to(hidden_states.dtype)
            module_output = self.module(module_input)[..., :seq_len] 


        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx]
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
                # conv_state = nn.functional.pad(hidden_states,(self.conv_kernel_size - hidden_states.shape[-1], 0))
                conv_state = nn.functional.pad(hidden_states,(self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state.clone()
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len]) # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros((batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len]) # [batch, intermediate_size, seq_len]
            
        # import pdb;pdb.set_trace()
        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)

        if  selective_scan_fn is not None and selective_state_update is not None:
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2) # [batch, seq_len, intermediate_size]
            if self.module and module_output is not None:
                if self.gated_conv1d:
                    discrete_time_step = discrete_time_step * self.act(self.gated_conv1d(module_output)[...,:seq_len])
                if self.directdecay:
                    discrete_time_step = discrete_time_step * create_distant_decay_matrix(seq_len).to(hidden_states.device) 

            A = -torch.exp(self.A_log.float()) # [intermediate_size, ssm_state_size]
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None

            if cache_params is not None and cache_params.seqlen_offset > 0:
                ssm_state = cache_params.ssm_states[self.layer_idx]
                scan_output = selective_state_update(
                    ssm_state,
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:

                scan_output, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
        else:
            discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
            discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]
            # plot1(discrete_time_step.mean(-1),type="plt")
            # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
            A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
            
            discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
            discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float() # [batch, intermediade_size, seq_len, ssm_state_size]
            deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

       
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            scan_outputs = []
            for i in range(seq_len):
                ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]  # [batch, intermediade_size, ssm_state]
                scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
                scan_outputs.append(scan_output[:, :, 0])
            scan_output = torch.stack(scan_outputs, dim=-1) # [batch, seq_len, intermediade_size]
            scan_output = scan_output + (hidden_states * self.D[None, :, None])
            scan_output = (scan_output * self.act(gate))

            if cache_params is not None:
                cache_params.ssm_states[self.layer_idx] = ssm_state.clone()

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2)) # [batch, seq_len, hidden_size]

        return contextualized_states


    def forward(self, hidden_states, cache_params=None, extra_kwargs=None):
     
        # if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
        #     # log_c("Using cuda_kernels_forward")
        #     return self.cuda_kernels_forward(hidden_states, cache_params)
            
        return self.slow_forward(hidden_states, cache_params)


class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx, conv1d_configs=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMixer(config, layer_idx=layer_idx, conv1d_configs=conv1d_configs)


    def forward(self, hidden_states, cache_params=None, extra_kwargs=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, cache_params=cache_params, extra_kwargs=extra_kwargs)
        hidden_states = residual + hidden_states
        return hidden_states 


class MambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MambaConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["MambaBlock"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                nn.init.constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                nn.init.uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = torch.exp(
                torch.rand(self.config.intermediate_size)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_proj.bias.copy_(inv_dt)
            module.dt_proj.bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_layers)


class CustomMambaModel(MambaPreTrainedModel):
    def __init__(self, config, conv1d_configs=None) -> None:
        super().__init__(config)
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.max_position_embeddings = conv1d_configs.get("max_position_embeddings",None)
        self.use_relative_position = conv1d_configs.get("use_relative_position", None)
        self.use_abs_position = conv1d_configs.get("use_abs_position", None)
        
        if self.use_abs_position:
            self.wpe = nn.Embedding(self.max_position_embeddings, config.d_model)
        elif self.use_relative_position:
            freqs = torch.exp(-np.log(10000.0) / config.d_model * torch.arange(0, config.d_model, 2).float())
            self.register_buffer('freqs', freqs)
        
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx, conv1d_configs=conv1d_configs) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
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
    ) -> Union[Tuple, MambaOutput]:
        
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
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        position_embeds = None
        input_shape = input_ids.shape
        if self.use_relative_position or self.use_abs_position:
            if position_ids is None:
                if cache_params is not None:
                    position_ids = torch.arange(cache_params.seqlen_offset, input_shape[-1] + cache_params.seqlen_offset, dtype=torch.long).to(input_ids.device)
                    position_ids = position_ids.unsqueeze(0)
                else:
                    position_ids = torch.arange(input_shape[-1], dtype=torch.long).to(input_ids.device)
                    position_ids = position_ids.unsqueeze(0)

            if self.use_abs_position:
                position_embeds = self.wpe(position_ids).to(inputs_embeds.dtype)
            elif self.use_relative_position:  # TODO: DEBUG
                freqs = self.freqs.float().to(position_ids.device)
                position_ids = position_ids.float()
                # Add a small constant to avoid division by zero
                freqs += 1e-7
                angles = position_ids.unsqueeze(-1) / freqs.unsqueeze(0)
                position_embeds = torch.cat([angles.sin(), angles.cos()], dim=-1).to(inputs_embeds.dtype)

        hidden_states = inputs_embeds + position_embeds if position_embeds is not None else inputs_embeds

        all_hidden_states = () if output_hidden_states else None

        
        for layer_idx, mixer_block in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params, extra_kwargs=extra_kwargs)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params, extra_kwargs=extra_kwargs)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class CustomMambaForCausalLM(MambaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, custom_conv1d_configs=None,) -> None:
        super().__init__(config)
        self.config = config
        
        self.backbone = CustomMambaModel(
            config, 
            conv1d_configs=custom_conv1d_configs,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[MambaCache] = None,
        extra_kwargs =None,
        **kwargs,
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
            }
        )
        # model_inputs["cache_params"] = cache_params
        model_inputs["extra_kwargs"] = extra_kwargs
        
        return model_inputs

    def forward(self, input_ids: Optional[torch.LongTensor] = None, position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, cache_params: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None, output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, **kwargs):
      
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_kwargs=kwargs,  # for analysis like depth and ctx_length
        )
        import pdb;pdb.set_trace()
        hidden_states = mamba_outputs[0]
            
        if kwargs.get('num_last_tokens'):
            num_last_tokens = kwargs['num_last_tokens']
            if num_last_tokens > 0:
                hidden_states = hidden_states[:, -num_last_tokens:]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        lm_loss, total_loss = None, None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
       
        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return MambaCausalLMOutput(
            loss=lm_loss,
            total_loss=total_loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )
    


if __name__ == "__main__":
     
    x = torch.randn(1, 4, 8, dtype=torch.float32).cuda() # same type as the input
    conv_state = nn.functional.pad(x,(-2, 0))
    