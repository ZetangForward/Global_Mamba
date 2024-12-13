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
        # import pdb;pdb.set_trace()
        # if self.cfg.task.task_name.lower()=="mqar":
        #     ctx_len, labels, num_kv_pairs, value_len = batch['ctx_length'], batch['labels'], batch['num_kv_pairs'], batch['value_len']
        #     target_len = int(num_kv_pairs * (value_len + 1) + 1)
        #     input_len = ctx_len - target_len
        #     output = self.model(input_ids).logits.max(-1)[1]
        #     return None
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

            # import pdb;pdb.set_trace()
            # self.params_for_debug_per_example.append(outputs.record_params)
            # if outputs.record_params is not None:
            #     outputs.record_params["ctx_length"] = final_res['ctx_length']
            #     outputs.record_params["num_kv_pairs"] = final_res['num_kv_pairs']
            #     outputs.record_params["quick_check"] = final_res['quick check']
            #     analysis_root_path = f"/nvme1/zecheng/analysis/MQAR/{self.cfg.experiment.experiment_name}/params"
            #     os.makedirs(analysis_root_path, exist_ok=True)
            #     analysis_name = f"ctx_{final_res['ctx_length']}-kvpairs_{final_res['num_kv_pairs']}.pt"
            #     analysis_path = os.path.join(analysis_root_path, analysis_name)
            #     torch.save([outputs.record_params], analysis_path)
        
        elif "ppl" in self.cfg.task.task_name.lower():
            output = self.model(input_ids).logits.max(-1)[1]

        elif "copy" in self.cfg.task.task_name.lower():
            labels = batch['labels']
            context_len = input_ids.shape[-1]
            max_gen_len = labels.shape[-1] + 10
            
            output = self.model.generate(input_ids, max_length=int(input_ids.size(-1) + max_gen_len),
                num_beams=1, do_sample=False, temperature=1.0, min_length=input_ids.size(-1) + labels.shape[-1],
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.encode("\n", add_special_tokens=False)[-1]])[0]
            
            final_res = {}
            final_res['predictions'] = output[context_len:]
            final_res['labels'] = labels

        elif "longbench" in self.cfg.task.task_name.lower():
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            max_gen_len, tag = batch['max_gen_len'].item(), batch['tag']
            output = self.model.generate(input_ids, attention_mask=attention_mask, min_new_tokens=1, max_new_tokens=max_gen_len)
            pred = self.tokenizer.decode(output[0][context_length:], skip_special_tokens=True)
            
            if batch['tag'] == "samsum": 
                output = self.model.generate(
                    input_ids,
                    max_length=int(input_ids.size(-1) + max_gen_len),
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = self.model.generate(
                    input_ids,
                    max_length=int(input_ids.size(-1) + max_gen_len),
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            
            final_res = {}
            final_res['predictions'] = pred
            final_res['labels'] = batch.pop('label')
            final_res['tag'] = batch.pop('tag')

            if batch.get("all_classes"):
                final_res['all_classes'] = batch.pop('all_classes')

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

    # input_ids = tokenizer("How are you", return_tensors="pt")["input_ids"].cuda()
    # model.eval()  # 切换到评估模式
    # out = model.generate(input_ids, max_new_tokens=10)
    # print(tokenizer.batch_decode(out))
    # import pdb;pdb.set_trace()
    
    
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

    if config.task.dataset.data_name == "pg19":
        evaluate_validation_set_ppl_test(model, tokenizer, data_module, config, 10, 1, )
        exit()
 
    # import pdb;pdb.set_trace()
    # if not os.path.exists(save_final_path):
    predictions = tester.predict(model=experiment, dataloaders=data_module.predict_dataloader(), return_predictions=True)
    # import pdb;pdb.set_trace()
    # if "ana" in config.experiment.experiment_name:
    #     log_c("FOR ANALYSIS, EXIT")
    #     exit()

    auto_save_data(predictions, save_final_path) 
    log_c("======= prediction end, begin to post process and save =======", "magenta")
    # else:
    #     log_c("predictions already exist, begin to evaluate")

    # import pdb;pdb.set_trace()
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

    # model_root_dir = config.platform.hf_model_path
    # save_root_dir = config.platform.result_path
    # data_root_dir = config.platform.dataset_path
    # model, tokenizer = get_model_tokenizer(model_root_dir, config, use_custom_module=config.model.use_custom_module)