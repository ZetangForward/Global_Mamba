import os
import sys
sys.path.append(os.getcwd())
import torch   
import hydra  
import transformers
import pytorch_lightning as pl
from custom_dataset.data import custom_datamodule
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from modelzipper.tutils import *


class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="eval") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = config
        self.tokenizer = tokenizer
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        
    def denormalize_func(self, normalized_tensor, min_val=0, max_val=200):
        tensor = (normalized_tensor + 1) / 2
        tensor = tensor * (max_val - min_val) + min_val
        tensor = torch.round(tensor).long()
        return tensor

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.model.generate(batch['input_ids'], max_new_tokens=64, temperature=0.9, top_p=0.7, eos_token_id=self.tokenizer.eos_token_id)
        label = batch['labels'][0]
        
        standard_test_reconstruct = {
            "prediction": self.tokenizer.decode(output[0]),
            "golden": self.tokenizer.decode(label),
        }
        
        return standard_test_reconstruct
    

@hydra.main(config_path='../../configs', config_name='test_gpt', version_base='1.1')
def main(config):
    
    print_c(f"Experiment: {config.experiment.task}", "magenta")
    
    # load model and tokenizer
    model = transformers.GPTNeoXForCausalLM.from_pretrained(config.model.model_name_or_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if "gpt-neo" in config.tokenizer.tokenizer_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        
    # load experiment (and model checkpoint)
    experiment = Experiment.load_from_checkpoint(config.model.ckpt_path, model=model, config=config, tokenizer=tokenizer)
    
    
    # load data
    data_module = custom_datamodule(config.dataset, tokenizer)

    tester = pl.Trainer(devices=config.experiment.device_num)

    b_t = time.time()

    predictions = tester.predict(
        experiment, 
        datamodule=data_module,
        return_predictions=True,
        ckpt_path=config.model.ckpt_path  # second pass for safety
    )
    
    print_c(f"======= prediction end, begin to post process and save =======", "magenta")

    save_path = f"{config.experiment.results_save_dir}/predictions.jsonl"
    auto_save_data(predictions, save_path)
    print_c(f"save predictions to {save_path}, total cost time: {time.time() - b_t}", "magenta")

if __name__ == '__main__':
    main()