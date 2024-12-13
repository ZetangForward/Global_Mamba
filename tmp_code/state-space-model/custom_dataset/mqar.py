############################################################
#                     MQAR DATASET                         #
############################################################

from typing import Any, Mapping, Tuple, List, Optional, Dict, Sequence, Union
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import torch
from torch import Tensor

class mqar_collate_fn:
    def __init__(self, pad_token_id, max_seq_length=None) -> None:
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch) -> Dict[str, Tensor]:
        max_ctx_length = max([item['ctx_length'] for item in batch])
        attention_masks = [torch.ones_like(item['input_ids']) for item in batch]
        input_ids = [torch.nn.functional.pad(item['input_ids'], (0, max_ctx_length - item['input_ids'].size(-1)), \
            mode='constant', value=self.pad_token_id) for item in batch]  # pad to the max_seq_length within the batch
        attention_masks = [torch.nn.functional.pad(item, (0, max_ctx_length - item.size(-1)), \
            mode='constant', value=0) for item in attention_masks]  # create attention mask
        labels = [torch.nn.functional.pad(item['labels'], (0, max_ctx_length - item['labels'].size(-1)), \
            mode='constant', value=-100) for item in batch]  # pad to the max_seq_length within the batch
        attention_masks, input_ids, labels = torch.stack(attention_masks), torch.stack(input_ids), torch.stack(labels)
        return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels, 
                "ctx_length": [item['ctx_length'] for item in batch], "num_kv_pairs": [item['num_kv_pairs'] for item in batch], \
                "kv_noise_len": [item['kv_noise_len'] for item in batch]}
    

class MQARDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", max_seq_length=None, *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.content = sorted(content, key=lambda x: x['ctx_length'], reverse=True)
        if max_seq_length is not None and split == 'valid': self.filter_length(2048)

    def filter_length(self, max_seq_length) -> List[Tensor]:
        new_content = [item for item in self.content if item['ctx_length'] <= max_seq_length]
        self.content = new_content

    def __len__(self) -> int:
        return len(self.content)
    
    def __getitem__(self, index) -> Dict[Tensor, int]:
        # import pdb;pdb.set_trace()
        item = self.content[index]
        ctx_length, input_ids, v_ids = len(item['input_ids']), item['input_ids'], item['V_id']
        num_kv_pairs = int(item['num_kv_pair'])
        input_ids = torch.LongTensor(input_ids)
        labels = torch.empty_like(input_ids).fill_(-100)
        fill_num = torch.LongTensor(list(v_ids.values()))
        fill_idx = torch.LongTensor([int(i) for i in list(v_ids.keys())]) # TODO: change all keys to int type
        labels = torch.scatter(labels, 0, fill_idx, fill_num)
        data = {"input_ids": input_ids, "labels": labels, "ctx_length": ctx_length, "num_kv_pairs": num_kv_pairs, 'key_len': 1, 'value_len': 1, 'kv_noise_len': -1}
        if 'key_len' in item.keys():
            data['key_len'] = item['key_len']
            data['value_len'] = item['value_len']
        if 'kv_noise_len' in item.keys():
            data['kv_noise_len'] = item['kv_noise_len']
        return data
    
        
if __name__ == '__main__':
    content = auto_read_data("/public/home/ljt/tzc/data/MQAR/train_based_tzc.jsonl")
    dataset = MQARDataset(content)
    dataloader = DataLoader(dataset, batch_size=14, collate_fn=custom_collate_fn(2))
    for item in dataloader:
        print(item)