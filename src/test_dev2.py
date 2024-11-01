import os  
import sys
sys.path.append(os.getcwd())
import torch   
import hydra  
import importlib
import lightning.pytorch as pl
from modelzipper.tutils import *
from utils import get_model_tokenizer, CustomDatamodule
from Custom_evaluate.Custom_evaluator import CustomEvaluator
from configs.config import parse_args, get_final_configs
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, enable_wrap, wrap
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from lightning.pytorch.plugins.precision import FSDPPrecision
from accelerate.big_modeling import dispatch_model, get_balanced_memory, infer_auto_device_map
from ppl_test import *



class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="eval", paras_for_debug_per_example=None) -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = config
        self.tokenizer = tokenizer
        self.params_for_debug_per_example = []

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids = batch.pop("input_ids")
        if "ar" in self.cfg.task.task_name.lower():
            # import pdb;pdb.set_trace()
            outputs = self.model(input_ids)
            output = outputs.logits.max(-1)[1]
            final_res = {}
            final_res['predictions'] = output[0]
            final_res['labels'] = torch.roll(batch['labels'], -1)
            final_res['ctx_length'] = int(batch.pop('ctx_length'))
            final_res['num_kv_pairs'] = int(batch.pop('num_kv_pairs'))
            final_res['input_ids'] = input_ids
            final_res['key_length'] = int(batch.pop('key_len'))
            final_res['value_length'] = int(batch.pop('value_len'))
            final_res['kv_noise_len'] = int(batch.pop('kv_noise_len'))

            label = final_res['labels'].squeeze(0)
            target_idx = label != -100
            pred_value = output[0].squeeze(0)[target_idx]
            label_value = label[target_idx]

            pred_value_chunk = torch.split(pred_value, final_res['value_length'])
            label_value_chunk = torch.split(label_value, final_res['value_length'])
      
            key_ = input_ids.squeeze(0)[torch.roll(target_idx, -1 * final_res['key_length'] + 1)]
            key_chunks = torch.split(key_, final_res['value_length'])
            key_chunks = [key[:final_res['key_length']] for key in key_chunks]

            tmp = []
            for i in range(len(key_chunks)):
                k, rv, pv = key_chunks[i], label_value_chunk[i], pred_value_chunk[i]
                tmp.append({"key": k.tolist(), "label_value": rv.tolist(), "pred_value": pv.tolist()})
            final_res['quick check'] = tmp
        elif "ppl" in self.cfg.task.task_name.lower():
            output = self.model(input_ids).logits.max(-1)[1]
        else:

            output = self.model.generate(
                input_ids, 
                max_length=input_ids.size(-1) + self.cfg.task.other_cfgs.max_generation_length,
                min_length=input_ids.size(-1) + 10, 
                eos_token_id=self.tokenizer.eos_token_id, 
            )
            final_res = {}
            final_res['predictions'] = output[0]

        return final_res
        

def main(config):
    model_root_dir = config.platform.hf_model_path
    save_root_dir = config.platform.result_path
    data_root_dir = config.platform.dataset_path

    pl.seed_everything(config.experiment.seed, workers=True)

    model, tokenizer = get_model_tokenizer(model_root_dir, config)
    print_c(model, "magenta")

    log_c("Current Experiment:" +  config.experiment.experiment_name)

    data_module = CustomDatamodule(config.task, data_root_dir, tokenizer)
    data_module.setup(stage='predict')
    
    # load experiment (and model checkpoint)
    experiment = Experiment(model=model, config=config, tokenizer=tokenizer)
    
    # init strategy
    log_c(f"utilize strategy {config.experiment.train_strategy}", "yellow")
    if config.experiment.train_strategy == "fsdp":
        dataset_module = importlib.import_module(config.experiment.model_module)
        NoSplitBlock = getattr(dataset_module, config.experiment.block_name)
        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={NoSplitBlock})
        strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, sharding_strategy='FULL_SHARD',
                                precision_plugin=FSDPPrecision(precision="bf16-true"), cpu_offload=True,
                                activation_checkpointing=NoSplitBlock)
    else: strategy = 'auto'

    tester = pl.Trainer(devices=config.experiment.device_num, strategy=strategy)
    
    b_t = time.time()


    save_path = os.path.join(save_root_dir, f"{config.task.dataset.data_name}/",f"{args.model_name_or_path}/",  f"{config.experiment.experiment_name}/")
    save_final_path = os.path.join(save_path, f"predictions.pkl")
    evaluation_path = os.path.dirname(save_path)

    predictions = tester.predict(model=experiment, dataloaders=data_module.predict_dataloader(), return_predictions=True)

    auto_save_data(predictions, save_final_path) 
    log_c("======= prediction end, begin to post process and save =======", "magenta")
   
    print_c(f"save predictions to {save_final_path}, total cost time: {time.time() - b_t}", "magenta")
    eval = CustomEvaluator(
        root_dir=save_root_dir, fpath=save_final_path, 
        data_path=data_root_dir,
        task=config.task.task_name,
        exp_name=config.experiment.experiment_name,
        tokenizer_name_or_path=None,
        value=None, save_evaluation_path=evaluation_path,
        save_gen_res=True,
    )
if __name__ == '__main__':

    args = parse_args()
    config = get_final_configs(args)
    print_c(config, 'yellow')
    main(config)

   