
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

global valid_loss_dict
valid_loss_dict = []


cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
if cuda_visible_devices is not None:
    device_num = len(cuda_visible_devices.split(','))
else:
    device_num = 1  # default to 1 if CUDA_VISIBLE_DEVICES is not set

class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="train", max_training_steps=None) -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.train()
        self.tokenizer = tokenizer
        self.cfg = config
        self.platform_cfg = config.platform
        self.max_training_steps = max_training_steps
        self.evaluator = None  # for validation set
        # import pdb;pdb.set_trace()
        # self.save_hyperparameters()
        
        if state == "train": self.loss_fct = torch.nn.CrossEntropyLoss()
        try: self.hold_graph = self.params['retain_first_backpass']
        except: pass
        
    def training_step(self, batch, batch_idx):
        # import pdb;pdb.set_trace()
        outputs = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        lm_loss = outputs.loss
        ppl = torch.exp(lm_loss)
    
        self.log("train_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        self.log("train_ppl", ppl, sync_dist=True,  prog_bar=False)

        self.last_batch_tokens = batch['input_ids'].numel()
        if not hasattr(self, 'total_tokens'):
            self.total_tokens = 0
        self.total_tokens += self.last_batch_tokens
        self.log('total_tokens', self.total_tokens, sync_dist=True, prog_bar=False)
        # print(self.model.state_dict()["backbone.layers.23.mixer.dt_proj.weight"])
        # import pdb;pdb.set_trace()
        return lm_loss
    def validation_step(self, batch, batch_idx):
        # import pdb;pdb.set_trace()
        outputs = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        lm_loss = outputs.loss
        ppl = torch.exp(lm_loss)
        self.log("valid_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        self.log("valid_ppl", ppl, sync_dist=True, prog_bar=True)
    def configure_optimizers(self):
        # import pdb;pdb.set_trace()
        num_warmup_steps = self.max_training_steps * 0.1
        num_training_steps = self.max_training_steps
        if self.cfg.optimizer.optimizer_type.lower() == "adamw":  # init optimizer
            optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.cfg.optimizer.lr,
                weight_decay=0.1,
                betas=(self.cfg.optimizer.beta_1, self.cfg.optimizer.beta_2),
            )
        else: # implement with adam as default 
            betas = (self.cfg.experiment.beta_1, self.cfg.experiment.beta_2)
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.cfg.experiment.peak_lr,
                weight_decay=self.cfg.experiment.weight_decay, 
                betas=betas, 
                eps=self.cfg.experiment.eps
            )
        
        # init lr scheduler
        if self.cfg.lr_scheduler.scheduler_type == "get_cosine_schedule_with_warmup":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            def get_scheduler(optimizer, num_training_steps, warmup_steps, peak_lr, last_lr):
                
                def lr_lambda(current_step):
                    if current_step < warmup_steps:
                        return current_step / warmup_steps
                    progress = (current_step - warmup_steps) / (num_training_steps - warmup_steps)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    lr = (last_lr + (peak_lr - last_lr) * cosine_decay)
                    return lr / peak_lr
                
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            scheduler = get_scheduler(
                optimizer, 
                self.cfg.optimizer.num_training_steps, 
                self.cfg.experiment.warmup_steps, 
                self.cfg.experiment.peak_lr, 
                self.cfg.experiment.last_lr
            )

        lr_scheduler = {
            'scheduler': scheduler,
            'name': f"{self.cfg.lr_scheduler.scheduler_type}",
            'interval': 'step',  # Ensure learning rate updates per step
            'frequency': 1,  # Optional: If you want to make sure it updates every step
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

class TokenCountCallback(Callback):
    def __init__(self, max_tokens):
        super().__init__()
        self.max_tokens = max_tokens
        self.total_tokens = 0

    def on_train_batch_end(self, trainer, pl_module,*args, **kwargs):
        # update token nums

        self.total_tokens += pl_module.last_batch_tokens * int(os.environ.get('WORLD_SIZE', 1))
        if self.total_tokens >= self.max_tokens:
            trainer.should_stop = True


def saint_check(config):  # TODO: failed for multi_nodes
    world_size = os.environ.get('WORLD_SIZE', None)
    if world_size is not None:
        world_size = int(world_size)
        log_c(f"Total number of processes (world size): {world_size}")
    else:
        log_c("WORLD_SIZE environment variable is not set.")

    if world_size is not None:
        if world_size != config.experiment.device_num:
            log_c(f"warning: num_device dose not match world size, change it to {world_size}")
            config.experiment.device_num = world_size

    return config

def main(config):
    model_root_dir = config.platform.hf_model_path
    save_root_dir = config.platform.exp_path
    data_root_dir = config.platform.dataset_path

    # import pdb;pdb.set_trace()
    def set_seed(seed=42):
        random.seed(seed)
        os.environ['PYHTONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    set_seed(config.experiment.seed)
    log_c(config.experiment.seed)
    # pl.seed_everything(config.experiment.seed, workers=True)
    

    # load model and tokenizer
    if config.experiment.low_rank_train:
        model, tokenizer = get_low_rank_model_tokenizer(model_root_dir, config.model)
    else:
        model, tokenizer = get_model_tokenizer(model_root_dir, config)
    print_c(model, "magenta")


    # load data
    data_module = CustomDatamodule(config.task, data_root_dir, tokenizer)
    data_module.setup(stage='fit')


    # calculate the training steps, epoches
    global_processes = device_num * config.experiment.node_num * config.experiment.accumulate_grad_batches 
    one_epoch_training_steps = len(data_module.train_dataloader()) // global_processes
    if config.experiment.max_training_steps is None:
        assert config.experiment.max_epochs is not None, "max_epoches must be defined !"
        total_training_steps = config.experiment.max_epochs * one_epoch_training_steps
    else: total_training_steps = config.experiment.max_training_steps
    log_c(f"global_batch_size(include grad accmulate): {global_processes * config.task.train_batch_size}")
    log_c(f"training steps per epoch: {one_epoch_training_steps}")
    log_c(f"total_training_steps: {total_training_steps}")

    # load experiment
    experiment = Experiment(model, config, tokenizer=tokenizer, state="train", max_training_steps=total_training_steps)

    # init logger
    tb_logger = TensorBoardLogger(save_dir=save_root_dir, name=config.experiment.experiment_name, version=config.experiment.version)
    log_c(f"using monitor:{config.experiment.monitor_metric}")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    train_lm_loss_ckpt_monitor = ModelCheckpoint(
        save_top_k=config.experiment.save_top_k, 
        dirpath =os.path.join(tb_logger.log_dir, "checkpoints"), 
        monitor="train_lm_loss",
        filename="{epoch}-{step}-{train_lm_loss:.2f}",
        save_last=True,
        mode='min',
        save_weights_only=True, # only save state dict
        every_n_train_steps=config.experiment.every_n_train_steps,
    )
    valid_lm_loss_ckpt_monitor = ModelCheckpoint(
        save_top_k=config.experiment.save_top_k, 
        dirpath =os.path.join(tb_logger.log_dir, "checkpoints"), 
        monitor="valid_lm_loss",
        filename="{epoch}-{step}-{valid_lm_loss:.2f}",
        save_last=True,
        mode='min',
        save_weights_only=True, # only save state dict
        every_n_train_steps=config.experiment.every_n_train_steps,
    )
    token_monitor = TokenCountCallback(max_tokens=5e10)  # max load 100b tokens

    # init strategy
    log_c(f"utilize strategy {config.experiment.train_strategy}", "yellow")
    if config.experiment.train_strategy == "fsdp":
        dataset_module = importlib.import_module(config.experiment.module)
        NoSplitBlock = getattr(dataset_module, config.experiment.class_name)
        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={NoSplitBlock})
        strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, sharding_strategy='FULL_SHARD',
                                precision_plugin=FSDPPrecision(precision="bf16-true"), cpu_offload=True,
                                activation_checkpointing=NoSplitBlock)
    elif config.experiment.train_strategy == "ddp": 
        strategy = DDPStrategy(find_unused_parameters=True)
    elif config.experiment.train_strategy == "deepspeed":
        strategy = DeepSpeedStrategy(stage=2, offload_optimizer=True, offload_params_device='cpu', offload_parameters=True, 
            partition_activations=True, contiguous_memory_optimization=False, cpu_checkpointing=True, logging_level=logging.INFO, 
            precision_plugin="bf16")
    else: strategy = 'auto'


    pl_trainer = Trainer(
        default_root_dir=os.path.join(tb_logger.log_dir, "checkpoints"),
        logger=tb_logger,
        callbacks=[lr_monitor, train_lm_loss_ckpt_monitor, valid_lm_loss_ckpt_monitor, token_monitor],
        check_val_every_n_epoch=1 if data_module.val_dataloader is not None else 1000000,  # set a large number if no validation set
        val_check_interval= 1.0 if config.experiment.val_check_interval is None else config.experiment.val_check_interval,
        strategy=strategy,
        max_steps=total_training_steps,
        precision='bf16',
        accumulate_grad_batches=config.experiment.accumulate_grad_batches,
        enable_checkpointing=True,
        devices='auto',
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm' if config.experiment.train_strategy == "deepspeed" else 'value',
        enable_model_summary=True,
        num_sanity_val_steps=1,
        fast_dev_run=5 if config.experiment.debug else False # for debugging
    )


    if config.experiment.train_strategy == "deepspeed": pl_trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    pl_trainer.print(torch.cuda.memory_summary())
    pl_trainer.fit(experiment, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())


if __name__ == '__main__':

    args = parse_args()
    config = get_final_configs(args)
    print_c(config, 'yellow')
    main(config)




