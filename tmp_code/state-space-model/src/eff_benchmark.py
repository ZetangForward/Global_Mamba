
import os
import sys
sys.path.append(os.getcwd())
import torch
import lightning.pytorch as pl
import logging
import functools
import torch
from builtins import hasattr
from torch import optim, Tensor 
from lightning.pytorch import Trainer
import importlib
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.precision import FSDPPrecision
from lightning.pytorch import Callback
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, enable_wrap, wrap
from configs.config import parse_args, get_final_configs
from modelzipper.tutils import *
from utils import *
import numpy as np
from evaluate.evaluator import *
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import trange
from torch.cuda import max_memory_allocated, memory_allocated


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'


class Profile(pl.LightningModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.batch_size = config.task.train_batch_size
        self.seq_len = config.task.dataset[0].max_seq_length

        self.mixed_precision = 'bf16'
        # Optimizer and scheduler
        model_root_dir = config.platform.hf_model_path
        save_root_dir = config.platform.exp_path
        data_root_dir = config.platform.dataset_path
        model, tokenizer = get_model_tokenizer(model_root_dir, config)
        self.model = model.to(device='cuda', dtype=torch.bfloat16)
        self.optimizer = AdamW(self.model.parameters())
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 0, 1024)
        # print(self.model)
        num_parameters = model.num_parameters()
        # print(f"Initializing {name} model from the config:\n{config}\n{model}")
        print(f"Number of parameters in total: {num_parameters} ({sizeof_fmt(num_parameters)})")
        print(f"Allocated memory after initialization: {sizeof_fmt(memory_allocated('cuda'))}")


        
    def forward(self, input_ids):
        return self.model(input_ids, labels=input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        outputs = self(input_ids)
        loss = outputs.loss
        return loss

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        # Create random dataset for profiling
        dataset = torch.utils.data.TensorDataset(
            torch.randint(0, self.model.config.vocab_size, size=(self.batch_size, self.seq_len), dtype=torch.long)
        )
        return DataLoader(dataset, batch_size=self.batch_size)

    def profile(self, warmup_steps: int = 16, steps: int = 100):
        print(f"batch_size: {self.batch_size}")
        print(f"seq_length: {self.seq_len}")
        # Warmup loop
        print(f"Warmup steps: {warmup_steps}")
        device = torch.device('cuda')
        torch.cuda.synchronize(device)
        for _ in trange(warmup_steps):
            tokens = torch.randint(high=self.model.config.vocab_size, size=(self.batch_size, self.seq_len)).cuda()
            outputs = self.model(tokens, labels=tokens)
            self.optimizer.zero_grad()
            outputs.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f"Max memory allocated: {sizeof_fmt(max_memory_allocated(device))}")

        # Profiling loop
        start_time = time.time()
        total_tokens = 0
        torch.cuda.synchronize(device)
        for _ in trange(steps):
            tokens = torch.randint(high=self.model.config.vocab_size, size=(self.batch_size, self.seq_len)).cuda()
            outputs = self.model(tokens, labels=tokens)
            self.optimizer.zero_grad()
            outputs.loss.backward()
            self.optimizer.step()

            total_tokens += self.batch_size * self.seq_len

        duration = time.time() - start_time
        throughput = total_tokens / duration
        print(f"Throughput: {throughput:.2f} tokens/s")

if __name__ == "__main__":
   
    args = parse_args()
    config = get_final_configs(args)
    print_c(config, 'yellow')
    # import pdb;pdb.set_trace()
    length = [2048, 4096, 8192, 16384]
    batch = [8, 4, 2, 1]
    for idx in range(0, len(length)):
        config.task.train_batch_size = batch[idx]
        config.task.dataset[0].max_seq_length = length[idx]
        p = Profile(config)
        p.profile()