import os  
import sys
sys.path.append(os.getcwd())
import torch   
import hydra  
import lightning.pytorch as pl
from modelzipper.tutils import *
from utils import get_model_tokenizer, CustomDatamodule
from evaluate.evaluator import Evaluator
from configs.config import parse_args, get_final_configs
from custom_mamba.custom_mamba_dev import CustomMambaForCausalLM

class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="eval") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = config
        self.tokenizer = tokenizer
        if hasattr(config.task, "inference_cfg"):  # what to save for task setting
            for key in config.task.inference_cfg:
                if isinstance(key, int):
                    key = str(key)
                setattr(self, key, config.task.inference_cfg[key])
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):

        input_ids = batch.pop("input_ids")

        depth = batch.get('depth').item()
        ctx_length = batch.get('before_insert_context_length').item()
        bos_pos, eos_pos = batch.get('bos_pos'), batch.get('eos_pos')

        if ctx_length % 1000 != 0:
            pass

        extra_kwargs = {
            "ctx_length": ctx_length,
            "depth": depth,
            "save_dir": "/nvme/zecheng/modelzipper/projects/state-space-model/analysis/inner_state2",
            "bos_pos": bos_pos, 
            "eos_pos": eos_pos,
        }
        
        output = self.model.generate(
            input_ids, 
            max_length=input_ids.size(-1) + self.cfg.task.other_cfgs.max_generation_length,
            min_length=input_ids.size(-1) + 10, 
            eos_token_id=self.tokenizer.eos_token_id, 
            extra_kwargs=extra_kwargs,
        )
        
        batch['predictions'] = output.squeeze(0)[input_ids.size(1):]
        batch['depth'] = depth
        batch['ctx_length'] = ctx_length
        batch['bos_pos'] = bos_pos
        batch['eos_pos'] = eos_pos
        
        return batch

       
def main():
    # if use_custom_module 
    ckpt_path = "/public/home/ljt/tzc/ckpt/longalpaca-long-context-pythia/version_1/checkpoints"
    
    model_path = "/public/home/ljt/hf_models/mamba-370m-hf"
    tokenizer_path = "/public/home/ljt/hf_models/mamba-370m-hf"
    conv1d_configs = {"kernel_sizes": [2, 4, 8, 16, 32, 64, 128, 256]}
    model = CustomMambaForCausalLM.from_pretrained(model_path, custom_conv1d_configs=conv1d_configs)
    # model = CustomMambaForCausalLM.from_pretrained(model_path).cuda()
    model.load_state_dict(original_model_weight)
    model = model.cuda()
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    dataset = auto_read_data("/nvme/zecheng/data/needle/processed_data/128k_500_insert_ids.pkl")

    for index in range(len(dataset)):
        item = dataset[index]
        tok_item = tokenizer(item['context_str'], return_tensors='pt').input_ids.cuda()

        depth = item.get('depth')
        ctx_length = item.get('before_insert_context_length')
        bos_pos, eos_pos = item.get('bos_pos'), item.get('eos_pos')

        if ctx_length % 1000 != 0:
            pass

        extra_kwargs = {
            "ctx_length": ctx_length,
            "depth": depth,
            "save_dir": "/nvme/zecheng/modelzipper/projects/state-space-model/analysis/inner_state2",
            "bos_pos": bos_pos, 
            "eos_pos": eos_pos,
        }
        
        output = model.generate(
            tok_item, 
            max_length=tok_item.size(-1) + 64,
            min_length=tok_item.size(-1) + 10, 
            eos_token_id=tokenizer.eos_token_id, 
            extra_kwargs=extra_kwargs,
        )

        res = tokenizer.batch_decode(output, skip_special_tokens=True)


    
if __name__ == '__main__':
    main()
