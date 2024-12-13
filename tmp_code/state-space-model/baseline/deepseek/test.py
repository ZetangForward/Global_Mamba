import os
import sys
sys.path.append(os.getcwd())
import torch   
import hydra 
import lightning.pytorch as pl
from modelzipper.tutils import *
from model import LlamaForCausalLM, LlamaModel
from utils import get_model_tokenizer, CustomDatamodule

class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, fpath=None, state="eval") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = config
        self.tokenizer = tokenizer
        self.f = open(fpath, "w")
        
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        
        output = self.model.generate(
            input_ids=batch.pop('input_ids'), 
            attention_mask=batch.pop('attention_mask'),
            max_new_tokens=64,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            output_attentions=True,
            return_dict=False,
            # depth = batch['depth'].cpu().item(),
            # ctx_length = batch['ctx_length'].cpu().item()
        )

        save_res = {
            "output": output.squeeze(0).cpu().tolist(),
            'depth': batch['depth'].item(),
            'ctx': batch['ctx_length'].item()
        }

        json_str = json.dumps(save_res)
        self.f.write(json_str + "\n")
        return save_res
    

@hydra.main(config_path='../../configs', config_name='test_config', version_base='1.1')
def main(config):
    
    print_c(OmegaConf.to_yaml(config), "yellow")

    model_root_dir = config.platform.hf_model_path
    save_root_dir = config.platform.result_path
    data_root_dir = config.platform.dataset_path

    # load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        os.path.join(model_root_dir, config.model.model_name_or_path), 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_root_dir, config.model.tokenizer_name_or_path))
    
    # load experiment (and model checkpoint)
    save_path = f"/nvme/zecheng/evaluation/passkey_search/deepseek-1_3b-16k/predictions.jsonl"
    experiment = Experiment(model=model, config=config, tokenizer=tokenizer, fpath=save_path)
    
    # load testing data
    data_module = CustomDatamodule(config.task, data_root_dir, tokenizer)
    data_module.setup(stage='predict')

    tester = pl.Trainer(devices=config.experiment.device_num)

    b_t = time.time()

    predictions = tester.predict(
        model=experiment,
        dataloaders=data_module.predict_dataloader(),
        return_predictions=True,
        ckpt_path=config.model.ckpt_path if config.model.load_model_state_dict else None
    )
    
    import pdb; pdb.set_trace()
    print_c(f"======= prediction end, begin to post process and save =======", "magenta")

    
    auto_save_data(predictions, save_path)
    print_c(f"save predictions to {save_path}, total cost time: {time.time() - b_t}", "magenta")

if __name__ == '__main__':
    main()