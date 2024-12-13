from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import glob


class LongAlignDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", full_modeling=True, max_seq_length=512, *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.max_text_length = max_seq_length
        self.tokenizer = tokenizer
        self.full_modeling = full_modeling
        self.template1 = "{instruction} {input} {output}"
        self.template2 = "{instruction} {output}"
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        sample = self.content[index]
        if 'input' not in sample.keys():
            context = self.template2.format(instruction=sample['instruction'], output=sample["output"])
        else:
            context = self.template1.format(instruction=sample['instruction'], input=sample["input"], output=sample["output"]) 
        
        tokenized_prompt = self.tokenizer(
            context,  
            truncation=True, 
            padding="max_length",
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = tokenized_prompt.input_ids[0]
        attention_mask = tokenized_prompt.attention_mask[0]
        labels = torch.where(
            input_ids != self.tokenizer.pad_token_id, input_ids, -100
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class LongAlignData(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.prepare_data_per_node = True
        self.dataset_kwargs = {
            "max_text_length": self.cfg.max_seq_length,
        }
        
    def setup(self, stage: str = 'fit') -> None:
        if self.cfg.inference_mode:
            pass
        else:
            content = auto_read_data(self.cfg.file_path)
            min_valid_num = min(1000, len(content)*0.1)
            self.valid_data = content[:min_valid_num]
            self.train_data = content[min_valid_num:]
            
            self.train_dataset = LongAlignDataset(
                content=self.train_data, 
                tokenizer=self.tokenizer, 
                split="train",
                max_seq_length=self.cfg.max_seq_length,
                **self.dataset_kwargs,
            )
            
            self.valid_dataset = LongAlignDataset(
                content=self.valid_data, 
                tokenizer=self.tokenizer, 
                split="valid",
                max_seq_length=self.cfg.max_seq_length,
                **self.dataset_kwargs,
            )
            print_c(f"num of train samples: {len(self.train_dataset)}", color='magenta')
            print_c(f"num of valid samples: {len(self.valid_dataset)}", color='magenta')

            
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.train_batch_size, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=True, shuffle=True, 
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.valid_dataset, batch_size=self.cfg.val_batch_size, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
        )
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return None
    