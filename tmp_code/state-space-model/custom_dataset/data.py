from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import glob
from datasets import load_dataset


class AlpacaDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", full_modeling=True, max_seq_length=512, *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.max_text_length = max_seq_length
        self.tokenizer = tokenizer
        self.full_modeling = full_modeling
        self.template = "{instruction} {input} {output}"
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        sample = self.content[index]
        instruction = sample["instruction"]
        input_text = sample["input"]
        output_text = sample["output"]
        
        prompt = self.template.format(instruction=instruction, input=input_text, output=output_text)
        
        tokenized_prompt = self.tokenizer(
            prompt,  
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


class TextFillingDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", full_modeling=True, *args, **kwargs):
        super(TextFillingDataset).__init__()
        self.split = split
        self.content = content
        self.max_text_length = kwargs['max_text_length']
        self.tokenizer = tokenizer
        self.full_modeling = full_modeling
        self.template1 = "Beginning: {s1} {s2} {s3}\nEnding: {s5}\nMiddle: "
        self.template2 = "Beginning: {s1} {s2} {s3}\nEnding: {s5}\nMiddle: {s4}"
        
    def __getitem__(self, index):
        sample = self.content[index]
        s1 = sample["sentence1"]
        s2 = sample["sentence2"]
        s3 = sample["sentence3"]
        s4 = sample["sentence4"]
        s5 = sample["sentence5"]
        
        if not self.full_modeling:
            prompt = self.template1.format(s1=s1, s2=s2, s3=s3, s5=s5)
            
            tokenized_prompt = self.tokenizer(
                prompt,  
                truncation=True, 
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            prompt_ids = tokenized_prompt.input_ids[0]
            label_ids = self.tokenizer(s4, return_tensors="pt").input_ids[0]
            
            if self.split == "test":  # test mode
                return {
                    "input_ids": prompt_ids,
                    "labels": label_ids,
                }
            
            prompt_mask = tokenized_prompt.attention_mask[0]
            prompt_sential = torch.empty_like(prompt_ids).fill_(self.tokenizer.pad_token_id)
            
            remain_length = self.max_text_length - prompt_ids.size(0)
            
            tokenized_mid = self.tokenizer(
                s4,  
                truncation=True, 
                padding="max_length",
                max_length=remain_length,
                return_tensors="pt",
            )
            label_ids = tokenized_mid.input_ids[0]
            label_attention_mask = tokenized_prompt.attention_mask[0]
            label_sentinel = label_ids
            
            input_ids = torch.concatenate([prompt_ids, label_ids], dim=0)
            tok_seq = torch.concatenate([prompt_sential, label_sentinel], dim=0)
            attention_mask = torch.concatenate([prompt_mask, label_attention_mask], dim=0)
            
            labels = torch.where(
                tok_seq != self.tokenizer.pad_token_id, tok_seq, -100
            )
        
        else:
            prompt = self.template2.format(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5)
            
            tokenized_prompt = self.tokenizer(
                prompt,  
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

    def __len__(self):
        return len(self.content)
    

class AlpacaData(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.prepare_data_per_node = True
        self.max_seq_length = self.cfg.max_seq_length
        
    def setup(self, stage: str = 'fit') -> None:
        self.test_dataset = None
        if self.cfg.inference_mode:
            pass
        else:
            content = auto_read_data(self.cfg.file_path)
            self.train_data = content
            self.train_dataset = AlpacaDataset(
                content=self.train_data, 
                tokenizer=self.tokenizer, 
                split="train",
                full_modeling=True,
                max_seq_length=self.max_seq_length,
            )
            print_c(f"num of train samples: {len(self.train_dataset)}", color='magenta')
      
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.train_batch_size, 
            num_workers=self.cfg.nworkers, 
            pin_memory=self.cfg.pin_memory, 
            drop_last=True, shuffle=True, 
        )

 
if __name__ == "__main__":
    file_path = "/nvme/zecheng/data/roc_stories/ROCStories_winter2017.csv"
    tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/gpt-neo-1.3B")
    data_module = custom_datamodule(file_path, tokenizer)
    raw_data = data_module.content
    import pdb; pdb.set_trace()