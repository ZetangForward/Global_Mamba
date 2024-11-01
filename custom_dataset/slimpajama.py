import torch
from torch.utils.data import Dataset
from modelzipper.tutils import *
import datasets
from itertools import chain

class Slimpajama(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs["max_seq_length"]

    def cluster_batch_fn(self):
        tmp = [item['source'] + ' ' + item['target'] for item in self.content]
        # tok_tmp = [self.tokenizer(item, return_tensors="pt") for item in tmp]
        sorted_tok_tmp = sorted(tmp, key=lambda x: len(x.split()))
        self.content = sorted_tok_tmp

    @classmethod
    def preprocess_data(cls, content, tokenizer, block_size, num_workers=1, column_names='text'):
        """
        (Pdb) content['train'][0].keys()
        dict_keys(['text', 'meta', '__index_level_0__'])
        """

        def tokenize_function(examples):
            res = tokenizer(examples[column_names])
            res.pop('attention_mask')
            return res

        def group_texts(examples):
            # Concatenate all texts.

            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()

            return result

        tokenized_datasets = content.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=['text', 'meta', '__index_level_0__'],
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        # import pdb;pdb.set_trace()
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=num_workers,
            load_from_cache_file=False,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        return lm_datasets

    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        # import pdb;pdb.set_trace()
        sample = self.content[index]
        input_ids = sample['input_ids']
        labels = sample['labels']
         
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }



