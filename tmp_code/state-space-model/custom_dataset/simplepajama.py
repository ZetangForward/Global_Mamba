from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import glob


class SimplepajamaDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs["max_seq_length"]
        self.cluster_batch = kwargs["cluster_batch"]

        if self.cluster_batch: 
            print_c("Requires clustering batch, begin to process", "yellow")
            bt = time.time()
            self.cluster_batch_fn()
            print_c(f"Clustering batch finished, time elapsed: {time.time()-bt}", "yellow")

    def cluster_batch_fn(self):
        tmp = [item['source'] + ' ' + item['target'] for item in self.content]
        # tok_tmp = [self.tokenizer(item, return_tensors="pt") for item in tmp]
        sorted_tok_tmp = sorted(tmp, key=lambda x: len(x.split()))
        self.content = sorted_tok_tmp

    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        if not self.cluster_batch:
            sample = self.content[index]
            src, tgt = sample['source'], sample['target']
            str_format = src + " " + tgt
        else: # after clustering batch, already in id format
            str_format = self.content[index]

        tokenized_sequence = self.tokenizer(  # re-tokenize to get attention mask
            str_format,  
            truncation=True, 
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = tokenized_sequence.input_ids[0]
        attention_mask = tokenized_sequence.attention_mask[0]
        labels = torch.where(
            input_ids != self.tokenizer.pad_token_id, input_ids, -100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
