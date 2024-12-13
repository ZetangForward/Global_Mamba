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

from modelzipper.tutils import * 
# from models.modules.mamba_analysis_utils import *
from models.modules.attention import *
from models.modules.longconv import *
from models.modules.mlp import *
from models.modules.flashconv import *
from models.modules.longconvlayer import *



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
    record_params:Optional[List[torch.FloatTensor]]=  None

@dataclass
class MambaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cache_params: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    record_params:Optional[List[torch.FloatTensor]]= None

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
        # self.long_conv_state = {
        #     i: torch.zeros(batch_size, intermediate_size, 512, device=device, dtype=dtype)
        #     for i in range(config.num_hidden_layers)
        # }

class FusionNetwork(nn.Module):
    def __init__(self, dimension):
        super(FusionNetwork, self).__init__()
        self.fc1 = nn.Linear(dimension * 2, dimension//4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dimension//4, dimension)

    def forward(self, tensor1, tensor2):
        concatenated = torch.cat((tensor1, tensor2), dim=1)
        concatenated = concatenated.transpose(1,2)
        fused_tensor = self.fc1(concatenated)
        fused_tensor = self.relu(fused_tensor)
        fused_tensor = self.fc2(fused_tensor)
        return fused_tensor.transpose(1,2)

class DT_linear(nn.Module):
    def __init__(self, dimension, time_step_rank):
        super(DT_linear, self).__init__()
        self.fc1 = nn.Linear(dimension , time_step_rank)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(time_step_rank, dimension)
    def forward(self, tensor):
        fused_tensor = self.act(tensor)
        fused_tensor = self.fc1(fused_tensor)
        fused_tensor = self.dropout(fused_tensor)
        fused_tensor = self.fc2(fused_tensor)
        return fused_tensor

class FuseShallow(nn.Module):
    def __init__(self, config):
        super(FuseShallow, self).__init__()
        dimension = config.intermediate_size
        self.proj = nn.Linear(dimension , dimension)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dimension//2, dimension //4),
            nn.SiLU(),
            nn.Linear(dimension//4, dimension)
        )
    def forward(self, hidden_state,  shallow_hidden_size, cur_x):
        # import pdb;pdb.set_trace()
        proj_shallow_h = self.proj(shallow_hidden_size.transpose(1,2))
        gate =  self.gate_mlp(cur_x)

        return hidden_state + (proj_shallow_h * gate).transpose(1,2)

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
        self.conv1d_configs=conv1d_configs

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
        self.save_params = None
        self.module_type = None

        if not conv1d_configs:
            if layer_idx==0: log_c(f"No module used, for custom forward")
        else:
            self.save_params = conv1d_configs.get("record_debug_params",None)
            # import pdb;pdb.set_trace()
            layer_list = [str(i) for i in range(config.num_hidden_layers)]
            if conv1d_configs.get("module_layers") and conv1d_configs['module_layers']:
                if "," in str(conv1d_configs['module_layers']):
                    layer_list = str(conv1d_configs['module_layers']).split(",")
                elif "-" in str(conv1d_configs['module_layers']):
                    layer_list = str(conv1d_configs['module_layers']).split("-")
                else:
                    layer_list = str(conv1d_configs['module_layers'])


            self.module_type = ""
            self.save_hidden = False
            if conv1d_configs.get("module_type") and "layer" not in conv1d_configs["module_type"]:
                self.module_type = ""
                if str(self.layer_idx) in layer_list:
                    self.module_type = conv1d_configs['module_type']
                    modules = conv1d_configs['module_type'].split("-")
                    for module in modules:
                        if module == "longconv" :
                            self.module = Longconv(config, layer_idx, conv1d_config=conv1d_configs)
                        elif module == "attn":
                            self.module = MHA(config, layer_idx, conv1d_configs)
                        elif module == "flashlongconv" or module=="shortlongconv":
                            if  "shallow" in self.module_type and str(self.layer_idx)==layer_list[0]:
                                pass
                            else:
                                self.module = FlashLongconv(config, layer_idx, conv1d_config=conv1d_configs)
                        elif module == "origin":
                            if layer_idx==0: log_c(f"No module used, for custom forward")
                        elif module == "shallow":
                            self.save_hidden = True
                            if str(self.layer_idx)!=layer_list[0]:
                                self.fusemodule = FuseShallow(config)
                        else:
                            if layer_idx==0: log_c(f"Wrong Module {module}", c='red')
            else: 
                if layer_idx==0: log_c("No module used, cuda_forward")


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
                cur_ssm_state = cache_params.ssm_states[self.layer_idx]   ### diff
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

       
        return contextualized_states, None

    def slow_forward(self, input_states, cache_params=None, saved_state=None, extra_kwargs=None):
        params_for_debug = {}
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2) # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)          # [batch, intermediate_size, seq_len]

        # if self.save_hidden and extra_kwargs and extra_kwargs.get("saved_hidden") is not None:
        #     # import pdb;pdb.set_trace()
        #     shallow_hidden_states = extra_kwargs["saved_hidden"]
        #     hidden_states = self.fusemodule(hidden_states, shallow_hidden_states, input_states)

        # if self.save_hidden:
        #     params_for_debug["saved_hidden"] = hidden_states

######################## Custom module sequence transformation   ##################################
        module_output = None
        if self.module and "longconv" in self.module_type and self.module_type!="shortlongconv":
            module_input = hidden_states.clone()
            module_input = module_input.to(hidden_states.dtype)
            module_output = self.module(module_input, cache_params).to(hidden_states.dtype)
        
##########################################################################################

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]


        module_act = None
        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        if self.module_type:
            if "+" in self.module_type and module_output is not None:
                module_act = self.act(module_output)   
                # module_act = module_output
                hidden_states = hidden_states + module_act
            if "concat" in self.module_type and module_output is not None:
                module_act = self.act(module_output)
                hidden_states = self.concat_net(hidden_states,module_act)
            if  self.module_type == "shortlongconv":
                hidden_states = self.module(hidden_states).to(hidden_states.dtype)
  

        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        A = -torch.exp(self.A_log.float()) # [intermediate_size, ssm_state_size]
        time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)

        # if  selective_scan_fn is not None and selective_state_update is not None:
        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)  # [batch, intermediate_size, seq_len ]
       

        if not self.training and self.save_params:
            params_for_debug['config'] = self.conv1d_configs
            params_for_debug['A'] = A.clone().float().cpu()
            params_for_debug['Sb_x'] = B.transpose(1,2).clone().float().cpu()  # B before discretization
            params_for_debug['C'] = C.transpose(1,2).clone().float().cpu()
            params_for_debug['dt_withoutsoftplus'] = (discrete_time_step.clone() + torch.broadcast_to(self.dt_proj.bias.clone().unsqueeze(dim=0).unsqueeze(dim=2), discrete_time_step.shape)).float().cpu()
            delta_t = F.softplus(discrete_time_step.clone() + torch.broadcast_to(self.dt_proj.bias.clone().unsqueeze(dim=0).unsqueeze(dim=2), discrete_time_step.shape))
            params_for_debug['delta_t'] = delta_t.clone().float().cpu()
            params_for_debug['B_t'] = None
            params_for_debug["module_act"] = module_act.clone().float().cpu() if module_act is not None else None 

        if self.module or module_output is not None:
            if "longconv" in self.module_type and "-" not in self.module_type and module_output is not None:  #and "+" not in self.module_type and "concat" not in self.module_type
                module_act =  self.act(module_output)
                discrete_time_step = discrete_time_step * module_act
            
            elif "attn" in self.module_type and "-" not in self.module_type and module_output is not None:
                module_act = self.act(module_output)
                discrete_time_step = discrete_time_step * module_act
            
            elif "dtlinear" in self.module_type  and "longconv" in self.module_type:
                module_act = self.dt_linear(module_output.transpose(1,2)).transpose(1,2)
                discrete_time_step = discrete_time_step + module_act

                if not self.training and self.save_params and module_act is not None:
                    params_for_debug["module_act"] = module_act.clone().float().cpu()
                    params_for_debug['dt_module_without_softplus'] = (discrete_time_step.clone() + torch.broadcast_to(self.dt_proj.bias.clone().unsqueeze(dim=0).unsqueeze(dim=2), discrete_time_step.shape)).float().cpu()
                    params_for_debug['dt_module_softplus'] = F.softplus(discrete_time_step.clone() + torch.broadcast_to(self.dt_proj.bias.clone().unsqueeze(dim=0).unsqueeze(dim=2), discrete_time_step.shape))
                    
            else: 
                discrete_time_step = discrete_time_step

       
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
            # import pdb;pdb.set_trace()
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
        
        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2)) # [batch, seq_len, hidden_size]

        return contextualized_states, params_for_debug


    def forward(self, hidden_states, cache_params=None, extra_kwargs=None):
        # import pdb;pdb.set_trace()
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type and (self.conv1d_configs is None or not self.conv1d_configs.get("module_type")):
        
            return self.cuda_kernels_forward(hidden_states, cache_params, extra_kwargs =extra_kwargs)
        
        return self.slow_forward(hidden_states, cache_params, extra_kwargs=extra_kwargs)

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

class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx, conv1d_configs=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMixer(config, layer_idx=layer_idx, conv1d_configs=conv1d_configs)

        self.module_layer=None
        if conv1d_configs and conv1d_configs.get("module_type") and  ("layer" in  conv1d_configs["module_type"]):

            module_number_list =   [str(i) for i in range(config.num_hidden_layers)]
            if conv1d_configs.get("module_layers") and conv1d_configs['module_layers']:
                module_number_list = conv1d_configs['module_layers'].split(",")

            if str(self.layer_idx) in module_number_list:
                self.module_layer = FlashLongconvLayer(config, layer_idx=self.layer_idx,\
                                                        conv1d_config=conv1d_configs,channels=1, \
                                                       dropout=0, kernel_learning_rate=0.001,\
                                                        kernel_lam=0.003, kernel_dropout=0.0)
    def forward(self, hidden_states, cache_params=None, extra_kwargs=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states, params_for_debug = self.mixer(hidden_states, cache_params=cache_params, extra_kwargs=extra_kwargs)
        hidden_states = residual + hidden_states
        if self.module_layer:
            hidden_states = self.module_layer(hidden_states)

        return hidden_states, params_for_debug
def init_params_for_debug(is_none_init=True):
    if is_none_init:
        return None
    
    params_for_debug = {}
    params_for_debug['A'] = []
    params_for_debug['Sb_x'] = []
    params_for_debug['C'] = []
    params_for_debug['delta_t'] = []
    params_for_debug['B_t'] = []
    params_for_debug['D'] = []

    params_for_debug['dt_withoutsoftplus'] = []
    params_for_debug["module_act"] = []
    params_for_debug['dt_module_without_softplus'] = [] 
    params_for_debug['dt_module_softplus'] = []

    return params_for_debug
def update_params_for_debug(params_for_debug, cur_params_for_debug, conv1d_configs):
    if conv1d_configs['record_debug_params']:
        if params_for_debug is not None and cur_params_for_debug is not None:
            # import pdb;pdb.set_trace()

            params_for_debug['A'].append(cur_params_for_debug['A'])
            params_for_debug['Sb_x'].append(cur_params_for_debug['Sb_x'])
            params_for_debug['C'].append(cur_params_for_debug['C'])
            params_for_debug['delta_t'].append(cur_params_for_debug['delta_t'])
            params_for_debug['B_t'].append(cur_params_for_debug['B_t'])
            params_for_debug['D'] = []

            params_for_debug['dt_withoutsoftplus'].append(cur_params_for_debug.get('dt_withoutsoftplus', None))
            params_for_debug["module_act"].append(cur_params_for_debug.get('module_act', None))
            params_for_debug['dt_module_without_softplus'].append(cur_params_for_debug.get('dt_module_without_softplus', None))
            params_for_debug['dt_module_softplus'].append(cur_params_for_debug.get('dt_module_softplus', None))

    return params_for_debug

class CustomMambaModel(MambaPreTrainedModel):
    def __init__(self, config, conv1d_configs=None) -> None:
        super().__init__(config)
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.conv1d_configs = conv1d_configs
        
   
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx, conv1d_configs=conv1d_configs) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

        if conv1d_configs and conv1d_configs.get("freeze_p") and conv1d_configs["freeze_p"]:
            log_c("Training Params")
            train_param = conv1d_configs["freeze_p"].split("-")

            for name, param in self.named_parameters():
                param.requires_grad = False
                for i in train_param:
                    if i in name:  
                        param.requires_grad = True
                        log_c(name)
                        break


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

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        record_params=None
        if self.conv1d_configs and self.conv1d_configs.get("record_debug_params"):
            record_params = init_params_for_debug(False)
        
        for layer_idx, mixer_block in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states, cur_params_for_debug = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params, extra_kwargs=extra_kwargs)
            else:
                hidden_states, cur_params_for_debug = mixer_block(hidden_states, cache_params=cache_params, extra_kwargs=extra_kwargs)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.conv1d_configs and self.conv1d_configs.get("record_debug_params"):
                record_params = update_params_for_debug(record_params, cur_params_for_debug, self.conv1d_configs)
            if cur_params_for_debug and  len(cur_params_for_debug) and  "saved_hidden" in cur_params_for_debug.keys():
                extra_kwargs["saved_hidden"] =  cur_params_for_debug["saved_hidden"]

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
            record_params=record_params,
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
        # self.save_hyperparameters()

        
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
        self, input_ids, cache_params: Optional[MambaCache] = None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache_params"] = cache_params
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
        hidden_states = mamba_outputs[0]

        record_params = mamba_outputs['record_params'] if 'record_params' in mamba_outputs else None
            
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
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
            record_params=record_params
        )
    


if __name__ == "__main__":
     
    x = torch.randn(1, 4, 8, dtype=torch.float32).cuda() # same type as the input
    conv_state = nn.functional.pad(x,(-2, 0))
    