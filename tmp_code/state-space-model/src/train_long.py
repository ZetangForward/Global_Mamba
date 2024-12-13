import os
from dataclasses import field, dataclass
from typing import Dict, Optional, Any, Union, List
import torch
from transformers import AutoTokenizer,TrainerCallback, TrainingArguments, TrainerState, TrainerControl,HfArgumentParser
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import sys
sys.path.append("/nvme1/zecheng/modelzipper/projects/state-space-model/custom_dataset")
sys.path.append("/nvme1/zecheng/modelzipper/projects/state-space-model")
from modelzipper.tutils import *
from transformers import Trainer
# from basedataset import BaseData
from models.long_pythia import GPTNeoForCausalLM
from longlora import LongLoRA
from accelerate.big_modeling import dispatch_model, get_balanced_memory, infer_auto_device_map
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig

@dataclass  
class CustomTrainingArguments(TrainingArguments):  
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training"}) 

@dataclass
class TrainingArguments(CustomTrainingArguments):
    model_name_or_path: Optional[str] = field(default="gpt2-xl")
    data_paths: List[str] = field(default_factory=lambda: ["./train.json"], metadata={"help": "Path to the training data."})
    instruction_length: int = 40
    output_length: int = 512
    save_steps = 60000
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_in_8bit: bool = field(default=True)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if torch.distributed.is_initialized():  
            is_main_process = torch.distributed.get_rank() == 0
        else:
            is_main_process = True 

        if is_main_process:

            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)

        return control

def train():

    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    print(args.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # if not tokenizer.pad_token_id:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    # max_seq_length = {
    #     "max_enc_length": 280,
    #     "max_dec_length": 400,
    # }
    
    # tokenizer_args = {
    #     "truncation": True,
    #     "padding": "max_length",
    #     "return_tensors": "pt",
    # }

    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
    # load model state dict
    config = GPTNeoConfig.from_pretrained("/nvme/hf_models/gpt-neo-1.3B")
    config.max_position_embeddings = 2048
    config.rope_scaling = None
    config.rope_theta = 10000.0
    config._attn_implementation = "flash_attention_2"
    config.chunk_attention = False

    model_max_length = 32768
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}
    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # instantiate model
    model = GPTNeoForCausalLM(config)
    raw_state_dict = transformers.GPTNeoForCausalLM.from_pretrained("/nvme/hf_models/gpt-neo-1.3B").state_dict()
    model.load_state_dict(raw_state_dict, strict=False)
    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    device_map = "balanced_low_0"
    no_split_module_classes = ["GPTNeoBlock"]
    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=no_split_module_classes,
        dtype=None,
        low_zero=(device_map=="balanced_low_0"),
    )
    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes, dtype=None
    )

    model = dispatch_model(model, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/gpt-neo-1.3B")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    raw_data = auto_read_data("/nvme1/zecheng/data/LongAlpaca-12k/LongAlpaca-12k.json")
    kwargs = {"max_seq_length":32768, "cluster_batch": False}
    dataset = LongLoRA(raw_data, tokenizer, **kwargs)
    # import pdb; pdb.set_trace()
    # model = GPT2LMHeadCLModel.from_pretrained(
    #     args.model_name_or_path,
    #     load_in_8bit=False,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    
    # lora hyperparams
    # config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=["c_attn"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    
    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()


    # dataset = BaseData(args.data_paths[0], tokenizer, tokenizer_args, max_seq_length, "train")
    model = model.to(torch.bfloat16)
    trainer = Trainer(
        model,
        args=args,
        # data_collator=dataset.collect_fn,
        train_dataset=dataset,
        callbacks=[SavePeftModelCallback],
    )

    trainer.train()

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train()