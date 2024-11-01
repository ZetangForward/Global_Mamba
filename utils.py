import torch
import os
import lightning.pytorch as pl
import importlib
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM,\
MambaConfig, AutoConfig, GPTNeoXForCausalLM, GPTJForCausalLM, AutoModel
from datasets import load_from_disk, load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from models.custom_mamba_v3 import CustomMambaForCausalLM
from models.custom_mamba_dev_dev_dev import CustomMambaForCausalLM as CustomMambaForCausalLMdevdev
from models.custom_mamba_v3_fast_dev import CustomMambaForCausalLM as CustomMambaForCausalLMdev
from accelerate.big_modeling import dispatch_model, get_balanced_memory, infer_auto_device_map
from modelzipper.tutils import *

from models.language_model import *
from models.model_config import *

class CustomModel(nn.Module):
    # for loading pytorch-lighting model ckpt
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

class EmptyDataset(Dataset):
    # for loading pytorch-lighting empty dataset
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise NotImplementedError

def long_context_pythia_model(model_path):
    # load model state dict
    config = GPTNeoConfig.from_pretrained(model_path)
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
    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  

    return model


def get_model_tokenizer_simple(root_dir, tokenizer_name_or_path=None, model_name_or_path=None):
    tokenizer, model = None, None
    if tokenizer_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(root_dir, tokenizer_name_or_path))
    if model_name_or_path is not None:
        model = AutoModelForCausalLM.from_pretrained(os.path.join(root_dir, model_name_or_path))

    return tokenizer, model

def get_low_rank_model_tokenizer(root_dir, model_config, use_custom_module=False):
    model_path = os.path.join(root_dir, model_config.model_name_or_path)
    tokenizer_path = os.path.join(root_dir, model_config.tokenizer_name_or_path)
    lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    # elif "mamba" in model_path.lower():
    model = CustomMambaForCausalLM.from_pretrained(
        model_path, use_relative_position=model_config.use_relative_position,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    for param in model.parameters():
        param.requires_grad = False
    peft_model = get_peft_model(model, lora_config, mixed=True)
    peft_model.print_trainable_parameters()
    return peft_model, tokenizer

def custom_from_pretrained(model, path, dtype, is_from_pytorch_lightning=False):  
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
    model.load_state_dict(state_dict, strict=False)
    return model

def get_model_tokenizer(root_dir, all_config, use_custom_module=False):
    model_config = all_config.model
    model_path = os.path.join(root_dir, model_config.model_name_or_path)
    tokenizer_path = os.path.join(root_dir, model_config.tokenizer_name_or_path)
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = f'cuda:{local_rank}'

 
    ############################################################
    #                        LOAD TOKENIZER                   #
    ############################################################
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    ############################################################
    #                   LOAD MODEL / CUSTOM MODEL              #
    ############################################################

    if "mamba" in model_path.lower():
        # import pdb;pdb.set_trace()
        if "tiny" in all_config.experiment.experiment_name: 
            log_c(f"Using CustomMambaForCausalLMfast, for all")
            log_c(f"tiny_mamba setting")
            config = MambaConfig.from_pretrained(model_path)
            if "tiny_mamba_config" in model_config:  # tiny mamba (2 layers)
                config.num_hidden_layers = model_config.tiny_mamba_config.num_hidden_layers
                config.time_step_rank = model_config.tiny_mamba_config.time_step_rank
                config.hidden_size = model_config.tiny_mamba_config.hidden_size
                config.intermediate_size = model_config.tiny_mamba_config.intermediate_size
                config.vocab_size = model_config.tiny_mamba_config.vocab_size
                config.state_size = model_config.tiny_mamba_config.ssm_state_size
            log_c(model_config)
            model = CustomMambaForCausalLMdev(config, custom_conv1d_configs=model_config.conv1d_configs).to(device)
        
        elif "lsgatedconv" in all_config.experiment.experiment_name.lower():
            log_c(f"Using CustomMambaForCausalLMfast, for all")
            config = MambaConfig.from_pretrained(model_path)
            model = CustomMambaForCausalLMconv(config, custom_conv1d_configs=model_config.conv1d_configs).to(device)

        else:
            log_c(f"Using CustomMambaForCausalLMfast, for all")
            config = MambaConfig.from_pretrained(model_path)
            if model_config.get("state_size", 16):
                if model_config.get("state_size"):
                    config.state_size = model_config.get("state_size", 16)
                if model_config.get("n_layers", 24):
                    config.num_hidden_layers = model_config.get("n_layers", 24)
            model = CustomMambaForCausalLMdev(config, custom_conv1d_configs=model_config.conv1d_configs).to(device)

    if "gpt-neo-125m" in model_path.lower():
        config = AutoConfig.from_pretrained(model_path)
        model = GPTNeoForCausalLM(config)
    
    if "pythia" in model_path.lower():
        # Load model directly
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        config = AutoConfig.from_pretrained("EleutherAI/pythia-160m")
        if "130m" in all_config.experiment.experiment_name.lower():
            config.num_hidden_layers=8  
        model = GPTNeoXForCausalLM(config)

    if "gla" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/pythia-160m")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        config = GLAConfig()
        config.bos_token_id = tokenizer.bos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id
        # import pdb;pdb.set_trace()
        config.attn_mode="chunk"
        config.vocab_size = tokenizer.vocab_size
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.tie_word_embeddings = True
        log_c(config)

        model = GLAForCausalLM(config).to(device).to(torch.bfloat16) 
 
    if "hgrn" in model_path.lower():
        config = HGRN2Config()
        config.bos_token_id = tokenizer.bos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.vocab_size = tokenizer.vocab_size
        config.num_hidden_layers = 12
        config.hidden_size = 768
        model = HGRN2ForCausalLM(config).to(device).to(torch.bfloat16)

    if "hyena" in model_path.lower():
        # import pdb;pdb.set_trace()
        config = ModelConfig()
        config.sequence_mixer = ModuleConfig(
            name="models.mixers.hyena.Hyena",
            kwargs={"l_max": 1024})
        config.state_mixer = ModuleConfig(
            name="models.mixers.mlp.MLP", 
            kwargs={"hidden_mult": 2}
        )
        config.d_model = 864
        config.n_layers = 18
        config.vocab_size = tokenizer.vocab_size
                   
        model = LanguageModel(config).to(device).to(torch.bfloat16)
    
    if "based" in model_path.lower():
        # import pdb;pdb.set_trace()
        config = ModelConfig()
        config.sequence_mixer = ModuleConfig(
            name="models.mixers.based.Based",
            kwargs={"l_max": 1024, 
                    "feature_dim": 16,
                    "feature_name": "taylor_exp",
                    "train_view": "quadratic",
                    "num_key_value_heads": 1,
                    "num_heads": 1,}
                    )
        config.state_mixer = ModuleConfig(
            name="models.mixers.mlp.MLP", 
            kwargs={"hidden_mult": 2})
        config.d_model = 1024
        config.n_layers = 12
        config.vocab_size = tokenizer.vocab_size
                   
        model = LanguageModel(config).to(device).to(torch.bfloat16)

    if "rwkv" in model_path.lower():
  
        config = ModelConfig()
        config.sequence_mixer = ModuleConfig(
            name="models.mixers.rwkv.RWKVTimeMixer",
            kwargs={"l_max": 1024})
        config.state_mixer = ModuleConfig(
            name="models.mixers.mlp.MLP", 
            kwargs={"hidden_mult": 2}
        )
        config.d_model = 1024
        config.n_layers = 12
        config.vocab_size = tokenizer.vocab_size
                   
        model = LanguageModel(config).to(device).to(torch.bfloat16)

    if model_config.ckpt_path is not None:
        def print_load(load_res):
            if load_res.missing_keys:
                log_c("Missing keys in state_dict:")
                log_c(load_res.missing_keys)
            if load_res.unexpected_keys:
                log_c("Unexpected keys in state_dict:")
                log_c(load_res.unexpected_keys)

        if model_config.ckpt_path == "hf":
            ckpt_path = model_path
            state_dict = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).state_dict()
        else:
            ckpt_path = model_config.ckpt_path
            state_dict = torch.load(model_config.ckpt_path, map_location='cpu')

            def convert_state_dict(state_dict):  
                if state_dict.get('state_dict'):
                    state_dict = state_dict['state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace('model.', '')] = v
                return new_state_dict
            state_dict = convert_state_dict(state_dict)

        log_c(f"loading model state dict from {ckpt_path}")
        load_res = model.load_state_dict(state_dict, strict=False)
        model = model.to(device).to(torch.bfloat16)
        print_load(load_res)

    return model, tokenizer


class CustomDatamodule(pl.LightningDataModule):

    def __init__(self, cfg, root_dir, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.prepare_data_per_node = True
    
    def load_data_with_root_dir(self, fpath, type='custom',cur_split="train"):
        if not self.root_dir in fpath:
            fpath = os.path.join(self.root_dir, fpath)
       
        if 'pg19' in fpath or 'pg19' in type :
            # import pdb;pdb.set_trace()
            try: return load_from_disk(fpath)[cur_split]
            except: return load_dataset(fpath)["test"]
        elif type == 'hf':  # TODO: FIXME (all datasets should load_from_disk)
            try: return load_from_disk(fpath)[cur_split]
            except: 
                try:return load_from_disk(fpath)
                except: return load_dataset(fpath)[cur_split]
        return auto_read_data(fpath)

    def setup(self, stage: str = 'fit') -> None:
        train_data, valid_data, test_data = None, None, None
        self.train_data_kwargs, self.valid_data_kwargs, self.test_data_kwargs = {}, {}, {}
        for data_cfg in self.cfg.dataset:  # we save training dataset and validation dataset configs respectively
            if stage == 'fit' and data_cfg.split == 'test': continue
            if stage != 'fit' and data_cfg.split in ['train', 'valid']: continue
            cur_split = data_cfg.split
            # import pdb;pdb.set_trace()
            # self.cfg.dataset = data_cfg     # TODO
            dataset_module = importlib.import_module(data_cfg.module)
            DatasetCLS = getattr(dataset_module, data_cfg.dataset_class_name)
            collect_fn_name = data_cfg.get("collate_fn_name", None)
            if collect_fn_name is not None: 
                Collect_fn = getattr(dataset_module, collect_fn_name)
                self.collate_fn = Collect_fn(max_seq_length=data_cfg.max_seq_length, pad_token_id=self.tokenizer.pad_token_id)
            else: self.collate_fn = None
            if data_cfg.inference_mode:  # load testing dataset
                self.cfg.dataset = data_cfg     # TODO
                if hasattr(data_cfg, "processed_data_path") and data_cfg.processed_data_path is not None:
                    test_data = self.load_data_with_root_dir(data_cfg.processed_data_path)
                else:
                    test_data = self.load_data_with_root_dir(data_cfg.data_path,type=data_cfg.type)
                self.test_data_kwargs.update({"max_seq_length": data_cfg.max_seq_length, "cluster_batch": data_cfg.cluster_batch, 
                                              "num_workers": data_cfg.nworkers, "pin_memory": data_cfg.pin_memory})
            else:  # load training dataset
                if hasattr(data_cfg, "processed_data_path") and data_cfg.processed_data_path is not None:
                    content = self.load_data_with_root_dir(data_cfg.processed_data_path,cur_split=cur_split)
                else:
                    assert "type" in data_cfg, "must define type in data_cfg ..."
                    if data_cfg.type.lower() in ["hf", "pg19"]: content = self.load_data_with_root_dir(data_cfg.data_path, type=data_cfg.type,cur_split=cur_split)
                    else: content = auto_read_data(os.path.join(self.root_dir, data_cfg.data_path))
                    
                extra_config = {"max_seq_length": data_cfg.max_seq_length, "cluster_batch": data_cfg.cluster_batch, 
                                "batch_size": data_cfg.batch_size,  "num_workers": data_cfg.nworkers, "pin_memory": data_cfg.pin_memory}
                    
                if data_cfg.split == "train": train_data = content; self.train_data_kwargs.update(extra_config)
                else: valid_data = content; self.valid_data_kwargs.update(extra_config)
            
            if stage == "fit":
                if data_cfg.split == "train": 
                    self.train_dataset = DatasetCLS(content=train_data, tokenizer=self.tokenizer, split="train", **self.train_data_kwargs)
                    print_c(f"num of train samples: {len(self.train_dataset)}", color='magenta')
                else:
                    # import pdb;pdb.set_trace()
                  
                    self.valid_dataset = DatasetCLS(content=valid_data, tokenizer=self.tokenizer, split="valid", **self.valid_data_kwargs)
                    print_c(f"num of valid samples: {len(self.valid_dataset)}", color='magenta')
            else: 
                assert test_data is not None, f"test data should not be None during {stage} stage"
                self.test_dataset = DatasetCLS(content=test_data, tokenizer=self.tokenizer, split="test",**self.test_data_kwargs)
                print_c(f"num of testing samples: {len(self.test_dataset)}", color='magenta')

        # saint check for if validation dataset is available
        if valid_data is None: self.valid_dataset = EmptyDataset(); print_c(f"No valid dataset, num of valid samples: 0", color='magenta')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # import pdb;pdb.set_trace()
        return DataLoader(self.train_dataset, batch_size=self.train_data_kwargs['batch_size'], 
                          num_workers=self.train_data_kwargs['num_workers'], pin_memory=self.train_data_kwargs['pin_memory'], 
                          drop_last=True, shuffle=False if self.train_data_kwargs['cluster_batch'] else True, 
                          collate_fn=self.collate_fn if self.collate_fn is not None else None)
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if isinstance(self.valid_dataset, EmptyDataset):
            return DataLoader(self.valid_dataset, num_workers=0)
        return DataLoader(self.valid_dataset, batch_size=self.valid_data_kwargs['batch_size'], 
                          num_workers=self.valid_data_kwargs['num_workers'], pin_memory=self.valid_data_kwargs['pin_memory'], 
                          drop_last=False, shuffle=False, collate_fn=self.collate_fn if self.collate_fn is not None else None)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_dataset is not None, "test dataset should not be None"
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.test_data_kwargs['num_workers'], \
                          pin_memory=self.test_data_kwargs['pin_memory'], drop_last=False, shuffle=False)


from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file
def load_state_dict_hf(model_name, device=None, dtype=None, cache_dir=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False, cache_dir=cache_dir)
    return torch.load(resolved_archive_file, map_location=mapped_device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict
