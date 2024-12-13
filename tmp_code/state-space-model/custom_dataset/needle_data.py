from modelzipper.datamanager import *
from modelzipper.tutils import *
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import glob


class CustomDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        """
        {"depth": depth, "context": context_insert, "needle": self.needle, "ctx_length": self.ctx_len}
        
        """
        sample = self.content[index]
        depth = sample["depth"]
        context = sample["context"]
        needle = sample["needle"]
        ctx_length = sample["ctx_length"]
        
        tokenized_context = self.tokenizer(context, return_tensors="pt")
        input_ids = tokenized_context.input_ids[0]

        return {
            "input_ids": input_ids,
            "depth": depth,
            "context": context,
            "needle": needle,
            "ctx_length": ctx_length,
        }
    

class FindNeedle(pl.LightningDataModule):

    def __init__(self, cfg, tokenizer, eval_path) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.eval_path = eval_path
        self.ctx_len = cfg.ctx_len
        self.needle = cfg.needle
        self.prepare_data_per_node = True

    def load_context(self, fpath, ctx_len=10000, tokenizer=None):
        context = ""
        for file in glob.glob(fpath):
            with open(file, 'r') as f: 
                context += f.read()
        tokenized_context = tokenizer(context, return_tensors="pt").input_ids
        tok_ids_len = len(tokenized_context[0])
        RATIO = len(context) / tok_ids_len
        context = context[: int(ctx_len * RATIO)]
        return context

    def insert_needle(self, context, needle, depth):
        context = context.split(".")
        c_len = len(context)
        needle_place = int(depth * c_len)
        context = ".".join(context[:needle_place]) + " ." + needle + ". ".join(context[needle_place:])
        return context

    def setup(self, stage: str = 'predict') -> None:
        all_insert_data = []
        context = self.load_context(fpath=self.eval_path, ctx_len=self.ctx_len, tokenizer=self.tokenizer)
        depth_list = [i * 0.05 for i in range(1, 21)]
        for i, depth in enumerate(depth_list):
            context_insert = self.insert_needle(context, self.needle, depth=depth)
            needle_idx = context_insert.find("The best thing to do in San Francisco is")
            print_c("Context has %d chars, needle inserted at %d char location:\n" % (len(context_insert), needle_idx), 'magenta')
            print_c(context_insert[needle_idx - 150: needle_idx + 150], 'cyan') # look at how the needle is inserted 
            print_c("-"*30)
            prompt ="\n<|im_start|> This is a very long story book: <book> %s </book>.\n" % context_insert
            question = "What is the best thing to do in San Francisco?"
            prompt += "Based on the content of the book, Question: %s\nAnswer:" % question
            all_insert_data.append({"depth": depth, "context": prompt, "needle": self.needle, "ctx_length": self.ctx_len})
        
        self.test_data = CustomDataset(all_insert_data, self.tokenizer, split="test")

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data, 
            batch_size=1, 
            num_workers=self.cfg.nworkers, 
            pin_memory=self.cfg.pin_memory, 
            drop_last=False, 
            shuffle=False,
        )
       