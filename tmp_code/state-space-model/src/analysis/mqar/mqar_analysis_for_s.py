import os 
import sys
sys.path.append(os.getcwd())
sys.setrecursionlimit(10000000)
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
from custom_mamba.custom_mamba_v3 import CustomMambaForCausalLM
from modelzipper.tutils import *
from utils import *
from dev_configs.config import parse_args, get_final_configs
from custom_dataset.AR_ywj import MQARDataset

class Conv1dAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = True

    def forward(self, weights, i):
        if self.use_flag:
            return self._forward(weights, i)
        else:
            return weights

    def _forward(self, weights, i):
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
        # self.params = nn.parameter.Parameter(torch.ones((cur_input_seq_len, 1, 512, cur_state_size)))
        # self.params.requires_grad_(True)
        
    def _forward(self, weights, i):
        if self.params is None:
            # self.params = torch.ones((cur_input_seq_len, 1, 512, cur_state_size), requires_grad=True).to(weights.dtype).to(weights.device)
            self.params = nn.parameter.Parameter(torch.ones((cur_input_seq_len, 1, 512, cur_state_size),device = weights.device)) #.to(weights.device)
            self.params.requires_grad_(True)
        # else:
            # self.params[i, :, :, :] = torch.ones_like(weights).to(weights.dtype)
        return weights * self.params[i, :, :, :].to(weights.device)
        # return weights * self.params

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
    projected_states = self.in_proj(input_states).transpose(1, 2) # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)
    # 2. Convolution sequence transformation
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx]
        if cache_params.seqlen_offset > 0:
            conv_state = cache_params.conv_states[self.layer_idx] # [batch, intermediate_size, conv_kernel_size]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = hidden_states[:, :, 0]
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1) # [batch, intermediate_size, 1] : decoding
        else:
            conv_state = nn.functional.pad(  # only save last conv_kernel_size states
                hidden_states,
                (self.conv_kernel_size - hidden_states.shape[-1], 0)
            )
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            hidden_states = self.conv1d(hidden_states)[:, :, :seq_len] # [batch, intermediate_size, seq_len]
            # tmp = adapter(tmp)
            hidden_states = self.act(hidden_states)     # [batch, intermediate_size, seq_len]

    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device, dtype=dtype
        )
        hidden_states = self.conv1d(hidden_states)[:, :, :seq_len] # [batch, intermediate_size, seq_len]
        # tmp = adapter(tmp)
        hidden_states = self.act(hidden_states) # [batch, intermediate_size, seq_len]

    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
    discrete_time_step = self.dt_proj(time_step) # [batch, seq_len, intermediate_size]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float() # [batch, intermediade_size, seq_len, ssm_state_size]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    # 3.c perform the recurrence y ← SSM(A, B, C)(x)
    scan_outputs = []
    for i in range(seq_len):
        tmp = adapter(ssm_state, i)
        ssm_state = discrete_A[:, :, i, :] * tmp + deltaB_u[:, :, i, :]  # [batch, intermediade_size, ssm_state]
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
            # hidden_states = adapter(hidden_states)
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
            
            # hidden_states = adapter(hidden_states)

            
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
            # if hasattr(self.model,"custom_conv1d_configs") and self.model.custom_conv1d_configs is not None and self.model.custom_conv1d_configs.kernel_sizes>4:
                # log_c("Using slow_forward,kernel_size>4")
            layer.mixer.slow_forward = partial(slow_forward, layer.mixer, adapter=conv1d_adapter)
            # else:
                # log_c("Using cuda_kernels_forward,kernel_size<=4")
                # layer.mixer.cuda_kernels_forward = partial(cuda_kernels_forward, layer.mixer, adapter=conv1d_adapter)

            conv1d_adapters.append(conv1d_adapter)
        return conv1d_adapters



def main(config):
    model_root_dir = config.platform.hf_model_path
    save_root_dir = config.platform.exp_path
    data_root_dir = config.platform.dataset_path

    pl.seed_everything(config.experiment.seed, workers=True)

    model, tokenizer = get_model_tokenizer(
        model_root_dir, config.model, 
        use_custom_module=config.model.use_custom_module
    )
    
    print_c(model, "magenta")    
    
    model.eval()

    len_kv = {
        '32' : [i for i in range(2,8,2)],                   
        '48' : [i for i in range(2,12,2)],                   
        '64': [i for i in range(2,16,2)],   # 64            
        '96': [i for i in range(2,24,2)],   # 64
        '128': [i for i in range(2,32,2)],  # 128
        '192': [i for i in range(2,48,2)],   # 64
        '224': [i for i in range(2,56,2)],   # 64
        '256': [i for i in range(2,64,2)],   # 256
        '384': [i for i in range(2,96,2)],   # 64
        '512': [i for i in range(2,128,2)]   # 512   
    }
    
    all_score = []
    cur_model = args.model_name_or_path +"-"+ args.experiment_name
    print(cur_model)
    for input_seq_len, kv_pairs in len_kv.items():
        
        global cur_input_seq_len
        global cur_state_size
        cur_input_seq_len = int(input_seq_len)
        cur_state_size =  int(args.model_name_or_path.split('-')[1][1:])
        print(cur_input_seq_len)
        print(cur_state_size)
        
        for kv_pair in kv_pairs:
            data_path = f"MQAR/analysis/test_C8192_N{input_seq_len}_D{kv_pair}.pkl"
            data_path = os.path.join(data_root_dir, data_path)
            if not os.path.exists(data_path):   
                test_data = MQARDataset.build_dataset(
                    vocab_size=8192, 
                    input_seq_len=int(input_seq_len),
                    num_kv_pairs=int(kv_pair),
                    num_examples=100,
                    power_a=0.01,
                    tokenizer=None,
                    insert_out_word=False,
                )
                auto_save_data(test_data, data_path)
            raw_data = auto_read_data(data_path)
            
            mean_mqar_score = [[],[]]
            sum_mqar_score = [[],[]]
            for data_id , data in tqdm(enumerate(raw_data)):
                conv1d_manger = Conv1dManager(model)
                conv1d_manger.zero_grad()  # TODO: important
                input_ids = data['input'].to(model.device).unsqueeze(0)
                output = model(input_ids=input_ids, extra_kwargs=None)
                label = data['label'].to(model.device).unsqueeze(0).long()
                output = output[0].view(-1, output[0].shape[-1])  # Reshape to (batch_size * seq_len, vocab_size)
                label = label.view(-1)
                loss = F.cross_entropy(output, label)            
                loss.backward(retain_graph=False)

                
                for layer_i in range(len(conv1d_manger.conv1d_adapters)):

                    saliency = conv1d_manger.grad(use_abs=True)[layer_i].squeeze().cpu()  # [seq_len, 512, statt_size] 
                    # import pdb;pdb.set_trace()
                    sum_score = torch.sum(saliency, dim=1)
                    mean_score = torch.mean(saliency, dim=1)
                    sum_mqar_score[layer_i].append(sum_score)
                    mean_mqar_score[layer_i].append(mean_score)

                conv1d_manger.zero_grad()  # TODO: important

            saliency = []
            for layer_idx in range(0,len(mean_mqar_score)):
                sum_score = sum_mqar_score[layer_idx]
                mean_score = mean_mqar_score[layer_idx]
                sum_score_stacked = torch.stack(sum_score)
                mean_score_stacked = torch.stack(mean_score)
                sum_score = torch.mean(sum_score_stacked, dim=0)    # 数据上取平均
                mean_score = torch.mean(mean_score_stacked, dim=0)    # 数据上取平均
                saliency.append({'sum_score':sum_score, 'mean_score': mean_score})
                # import pdb;pdb.set_trace()
            
            analysis_score = {
                "model" : cur_model  ,
                "input_seq_len" : input_seq_len,
                "kv_pairs" : kv_pair,
                "saliency" : saliency
            }
            # import pdb;pdb.set_trace()
            print(analysis_score)
            all_score.append(analysis_score)
    auto_save_data(all_score, f"/public/home/ljt/tzc/evaluation/analysis/MQAR/state_size/{cur_model}_state_adapter_score.pkl")


if __name__ == '__main__':
    args = parse_args()
    config = get_final_configs(args)
    print_c(config, 'yellow')
    import pdb;pdb.set_trace()
    
    main(config)
    