from modelzipper.tutils import *
from torch.utils.data import Dataset
import torch


class LongLoRA(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs["max_seq_length"]
        self.cluster_batch = kwargs["cluster_batch"]
        if self.cluster_batch:
            self.cluster_batch_fn()

    def cluster_batch_fn(self):
        # tok_tmp = [self.tokenizer(item, return_tensors="pt") for item in tmp]
        sorted_tok_tmp = sorted(self.content, key=lambda x: len(x['instruction'].split()), reverse=True)
        self.content = sorted_tok_tmp

    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        item = self.content[index]

        # Add a bos token at the beginning of instruct_input_ids
        instruct_input_ids = [self.tokenizer.bos_token_id] + self.tokenizer(item["instruction"] + " ", return_tensors="pt")["input_ids"][0].tolist()
        output_ids = self.tokenizer(item["output"], return_tensors="pt", )["input_ids"][0].tolist()

        total_length = len(instruct_input_ids) + len(output_ids)

        # Check if the total length exceeds max_seq_length
        if total_length > self.max_seq_length:
            # Calculate the excess length
            # Calculate the excess length based on the sum of instruct_input_ids and output_ids lengths
            excess_length = total_length - self.max_seq_length

            # Reduce the length of instruct_input_ids and output_ids based on their lengths ratio
            total_ids_length = len(instruct_input_ids) + len(output_ids)
            instruct_input_ids = instruct_input_ids[:int(len(instruct_input_ids) - len(instruct_input_ids) / total_ids_length * excess_length)]
            output_ids = output_ids[:int(len(output_ids) - len(output_ids) / total_ids_length * excess_length)]

        # Check if the total length is still more than max_seq_length due to rounding, if so, remove one more token from output_ids
        if len(instruct_input_ids) + len(output_ids) > self.max_seq_length:
            output_ids = output_ids[:-1]

        # Combine instruct_input_ids and output_ids
        input_ids = instruct_input_ids + output_ids

        return {"input_ids": torch.LongTensor(input_ids)}
    

class custom_collate_fn:
    def __init__(self, max_seq_length=64000, pad_token_id=1):
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch_max_length = min(self.max_seq_length, max([len(item["input_ids"]) for item in batch]))
        input_ids = [item["input_ids"][:batch_max_length] for item in batch]
        input_ids = [torch.nn.functional.pad(input=item, pad=(0, batch_max_length - item.size(-1)), mode='constant', value=self.pad_token_id) 
                    for item in input_ids]
        labels = [torch.where(item == self.pad_token_id, -100, item) for item in input_ids]
        return {"input_ids": torch.stack(input_ids, dim=0), "labels": torch.stack(labels, dim=0)}