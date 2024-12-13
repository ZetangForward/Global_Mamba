import os 
import sys
sys.path.append(os.getcwd())
import torch
import hydra
import torch.nn as nn
from functools import wraps, partial
from transformers import PreTrainedModel
from typing import Dict
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from custom_mamba.custom_mamba_v2 import CustomMambaForCausalLM
from modelzipper.tutils import *


class Conv1dAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = True

    def forward(self, weights):
        if self.use_flag:
            return self._forward(weights)
        else:
            return weights

    def _forward(self, weights):
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class Conv1dManagerBase:
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.conv1d_adapters = self.register_conv1d_to_model()
        self.model.forward = manager_decoractor(self)(self.model.forward)

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids
        for conv1d_adapter in self.conv1d_adapters:
            conv1d_adapter.register_input_ids(input_ids)

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_conv1d_to_model(self):
        raise NotImplementedError

    def zero_grad(self,set_to_none=True):
        if set_to_none:
            for conv1d_adapter in self.conv1d_adapters:
                conv1d_adapter.params = None
        else:
            for conv1d_adapter in self.conv1d_adapters:
                conv1d_adapter.zero_grad(set_to_none=True)

    def grad_process(self, grad, use_abs = False, use_dist = False):
        # assert len(grad.shape) == 4
        # grad = grad.sum(1)
        if use_abs:
            grad = abs(grad)
        return grad

    def grad(self,*args,**kwargs):
        grads = []
        for conv1d_adapter in self.conv1d_adapters:
            grads.append(self.grad_process(conv1d_adapter.params.grad,*args,**kwargs))
        return grads


def manager_decoractor(manager: Conv1dManagerBase):
    
    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None:
                input_ids = args[0]
            manager.register_input_ids(input_ids)
            return fn(*args, **kwargs)

        return wrapper

    return model_forward_decorator


class Conv1dAdapter(Conv1dAdapterBase):
    def __init__(self) -> None:
        super().__init__()
        self.params = None

    def _forward(self, weights):
        if self.params is None:
            self.params = torch.ones_like(weights, requires_grad=True).to(weights.dtype)
        else:
            self.params.data = torch.ones_like(weights).to(weights.dtype)
        return weights * self.params

    @property
    def grad(self):
        return self.params.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.params.grad is not None:
            if set_to_none:
                self.params.grad = None
            else:
                self.params.grad = torch.zeros_like(self.params.grad)


def slow_forward(self, input_states, cache_params=None, extra_kwargs=None, adapter=None):
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)

    # 2. Convolution sequence transformation
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx]
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
            conv_state = nn.functional.pad(  # only save last conv_kernel_size states
                hidden_states,
                (self.conv_kernel_size - hidden_states.shape[-1], 0)
            )
            cache_params.conv_states[self.layer_idx].copy_(conv_state)

            tmp = self.conv1d(hidden_states)[:, :, :seq_len] # [batch, intermediate_size, seq_len]
            tmp = adapter(tmp)
            hidden_states = self.act(tmp)     # [batch, intermediate_size, seq_len]
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device, dtype=dtype
        )
        tmp = self.conv1d(hidden_states)[:, :, :seq_len] # [batch, intermediate_size, seq_len]
        tmp = adapter(tmp)
        hidden_states = self.act(tmp)         # [batch, intermediate_size, seq_len]

    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
    )
    discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())                                             # [intermediate_size, ssm_state_size]
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    # 3.c perform the recurrence y ← SSM(A, B, C)(x)
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
        scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        scan_outputs.append(scan_output[:, :, 0])
    scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = (scan_output * self.act(gate))

    if cache_params is not None:
        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))             # [batch, seq_len, hidden_size]
    return contextualized_states



def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params=None, extra_kwargs=None, adapter=None):

    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(hidden_states).transpose(1, 2)

    if self.training and cache_params is None:  # Doesn't support outputting the states -> used for training
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
            
            hidden_states = adapter(hidden_states)
            
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
            scan_outputs = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
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


class Conv1dManager(Conv1dManagerBase):
    def __init__(self, model: PreTrainedModel):
        super().__init__(model)

    def register_conv1d_to_model(self):
        conv1d_adapters = []
        for i, layer in enumerate(self.model.backbone.layers):
            conv1d_adapter = Conv1dAdapter()
            # layer.mixer.cuda_kernels_forward = partial(cuda_kernels_forward, layer.mixer, adapter=conv1d_adapter)
            layer.mixer.slow_forward = partial(slow_forward, layer.mixer, adapter=conv1d_adapter)
            conv1d_adapters.append(conv1d_adapter)
        return conv1d_adapters


@hydra.main(config_path='../configs/platform', config_name='h_800', version_base='1.1')
def main(config):
    print_c(OmegaConf.to_yaml(config), "yellow")
    model_root_dir = config.hf_model_path
    exp_path = config.exp_path
    save_root_dir = config.result_path
    data_root_dir = config.dataset_path

    model_path = os.path.join(model_root_dir, "mamba-1.4b-hf")
    model = transformers.MambaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to('cuda:0')

    # model = CustomMambaForCausalLM.from_pretrained(model_path).to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    raw_data = auto_read_data(os.path.join(data_root_dir, "needle/processed_data/128k_500_insert_ids.pkl"))
    
    conv1d_manger = Conv1dManager(model)
    
    all_score = []
    model.eval()

    for idx, data in tqdm(enumerate(raw_data)):
        token_ids = data['context_ids']
        bos_pos, eos_pos = data['bos_pos'], data['eos_pos']
        ctx_length = data['before_insert_context_length']
        depth = data['depth']
        if ctx_length > 1000:
            break

        log_c(f"processing context length: {ctx_length}, depth {depth}", "yellow")
        
        per_ctx_length_score = []

        # token_ids = tokenizer(data['passkey_context'], return_tensors="pt").input_ids[0]
        input_ids = token_ids.to(model.device).unsqueeze(0)
        conv1d_manger.zero_grad()
        
        output = model(input_ids=input_ids, extra_kwargs=None)
        
        label = input_ids[:, -1].long()
        loss = F.cross_entropy(output[0][:, -2, :], label)
       
        loss.backward(retain_graph=True)

        for i in range(len(conv1d_manger.conv1d_adapters)):
            saliency = conv1d_manger.grad(use_abs=False)[i].squeeze()  # 4096, 532
            important_place_score = saliency[:, bos_pos: eos_pos].sum()
            other_place_score = saliency[eos_pos:].sum()
            proportion = important_place_score / other_place_score  # the larger the better
            per_ctx_length_score.append({"proportion": proportion, "depth": depth})

        all_score.append(per_ctx_length_score)

    auto_save_data(all_score, "/nvme/zecheng/evaluation/analysis/conv1d_adapter_score.pkl")


if __name__ == '__main__':
    main()
   

    