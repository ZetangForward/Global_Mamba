from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
import glob
from datasets import load_from_disk


class LongBenchDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", max_seq_length=None, *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def filter_length(self, max_ctx_length=12000):
        new_content = []
        print_c(f"begin to filter the context length | total {len(self.content)} instances", "yellow")
        for item in self.content:
            if item['ctx_length'] <= max_ctx_length:
                new_content.append(item)
        new_content = sorted(new_content, key=lambda x: x['ctx_length'], reverse=True)  # from long to short
        self.content = new_content
        print_c(f"filtering finished | total {len(self.content)} instances", "yellow")

    def cluster_batch_fn(self):
        tmp = [item['source'] + ' ' + item['target'] for item in self.content]
        sorted_tok_tmp = sorted(tmp, key=lambda x: len(x.split()))
        self.content = sorted_tok_tmp

    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        item = self.content[index]
        str_input, label, max_gen_len, tag = item['input'], item['label'], item['max_gen_len'], item['tag']
        tok_ipt = self.tokenizer(str_input, return_tensors="pt").input_ids
        if tok_ipt.size(-1) > self.max_seq_length:  # cut the sequence from the middle
            tok_ipt = torch.cat([tok_ipt[:self.max_seq_length//2], tok_ipt[-self.max_seq_length//2:]])
        if tok_ipt.dim() == 2: tok_ipt = tok_ipt.squeeze(0)
        attention_mask = torch.zeros_like(tok_ipt)
        return {"input_ids": tok_ipt, "labels": label, "attention_mask":attention_mask,
                 "max_gen_len": max_gen_len, "tag": tag}
