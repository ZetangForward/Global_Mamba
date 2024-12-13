from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import glob
from datasets import load_dataset


class ZeroScrollDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", max_seq_length=512, *args, **kwargs):
        super(ZeroScrollDataset).__init__()
        self.split = split
        self.max_text_length = max_seq_length
        self.tokenizer = tokenizer
        self.post_process(content)
        
    def post_process(self, content):
        self.content = []
        for key in content:
            for item in content[key]:
                self.content.append({"input_ids": item, "subset": key})
        
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        sample = self.content[index]
        return {"input_ids": sample["input_ids"], "subset": sample["subset"]}


class ZeroScrolls(pl.LightningDataModule):

    datasets = [
        'gov_report',
        'summ_screen_fd',
        'qmsum',
        'qasper',
        'narrative_qa',
        'quality',
        'musique',
        'squality',
        'space_digest',
        'book_sum_sort'
    ]

    def __init__(self, cfg, platform_cfg, tokenizer):
        super().__init__()
        self.data_cfg = cfg
        self.platform_cfg = platform_cfg
        self.tokenizer = tokenizer
        self.max_input_length = cfg.ctx_len
        self.prepare_data_per_node = True

    def trim_doc_keeping_suffix(self, tokenizer, tokenized_input_full, example, suffix_index, max_tokens, device):
        seperator_and_suffix = f"{example['truncation_seperator'].strip()}\n\n{example['input'][suffix_index:].strip()}\n"
        tokenized_seperator_and_suffix = tokenizer(seperator_and_suffix, return_tensors="pt").input_ids.to(device)
        tokenized_input_trimmed = tokenized_input_full[:, :max_tokens - tokenized_seperator_and_suffix.shape[1]]
        tokenized_input = torch.cat([tokenized_input_trimmed, tokenized_seperator_and_suffix], dim=1)
        return tokenized_input

    def process_model_input(self, tokenizer, example, max_tokens, device):
        tokenized_input_full = tokenizer(example["input"], return_tensors="pt").input_ids.to(device)
        if tokenized_input_full.shape[1] <= max_tokens:
            return tokenized_input_full
        seperator_and_query_text = example['truncation_seperator'] + example["input"][example['query_start_index']:]
        tokenized_seperator_and_query = tokenizer(seperator_and_query_text, return_tensors="pt").input_ids.to(device)
        input_without_query = example['input'][:example['query_start_index']]
        tokenized_input_without_query = tokenizer(input_without_query, return_tensors="pt").input_ids.to(device)
        tokenized_input_without_query = tokenized_input_without_query[:, :max_tokens - tokenized_seperator_and_query.shape[1]]
        tokenized_input = torch.cat([tokenized_input_without_query, tokenized_seperator_and_query], dim=1)
        return tokenized_input
    
    def setup(self, stage: str = 'predict') -> None:
        if self.data_cfg.processed_data_path is not None:
            all_testing_data = auto_read_data(os.path.join(self.platform_cfg.dataset_path, self.cfg.processed_data_path))
        else:
            all_testing_data = dict()
            print_c("processing data ...", "magenta")
            for dataset in self.datasets:
                print_c(f"processing split {dataset}", "magenta")
                all_testing_data[dataset] = []
                local_data_path = os.path.join(self.platform_cfg.dataset_path, self.data_cfg.data_path, dataset) # we save the data in local path
                data = load_dataset(local_data_path, split='test')
                for i, example in enumerate(data):
                    model_input = self.process_model_input(self.tokenizer, example, self.max_input_length, 'cpu')
                    all_testing_data[dataset].append(model_input)
            all_testing_data = all_testing_data
        
        self.test_data = ZeroScrollDataset(
            content=all_testing_data, 
            tokenizer=self.tokenizer, 
            split="test", 
            max_seq_length=self.max_input_length
        )

        
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data,
            batch_size=1,
            num_workers=self.cfg.nworkers,
            pin_memory=self.cfg.pin_memory, 
            drop_last=False, 
            shuffle=False,
        )
