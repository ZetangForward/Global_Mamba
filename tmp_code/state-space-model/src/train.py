
import os
import sys
sys.path.append(os.getcwd())
import torch
import lightning.pytorch as pl
import hydra
import logging
import torch
from builtins import hasattr
from torch import optim, Tensor 
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from modelzipper.tutils import *
from utils import *
from lightning.pytorch import Callback

class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="train") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.train()
        self.tokenizer = tokenizer
        self.cfg = config
        self.platform_cfg = config.platform

        if state == "train":
            self.loss_fct = torch.nn.CrossEntropyLoss()

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    def training_step_hf(self, batch, batch_idx):
        outputs = self.model(**batch)
        lm_loss = outputs.loss
        ppl = torch.exp(lm_loss)
        return lm_loss, ppl

    def validation_step_hf(self, batch, batch_idx):
        
        outputs = self.model(**batch)
        lm_loss = outputs.loss
        ppl = torch.exp(lm_loss)
        return lm_loss, ppl
      

    def training_step(self, batch, batch_idx):
        if self.cfg.experiment.hf_trainer:
            lm_loss, ppl = self.training_step_hf(batch, batch_idx)

        else:
            input_ids = batch["input_ids"]
            lm_logits = self.forward(input_ids).logits
            if "mqar" in self.cfg.task.dataset.class_name.lower():
                log_c("mqar_trainging")
                labels = batch.pop("label")
                labels = labels.long()
                shift_logits = lm_logits.contiguous()
                labels = labels.contiguous()

            else:
                labels = batch.pop("labels")
                shift_logits = lm_logits[:, :-1, :].contiguous()
                labels = labels[:, 1:].contiguous()

            labels = labels.to(lm_logits.device)

            lm_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            ppl = torch.exp(lm_loss)
            
        self.log("train_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        self.log("train_ppl", ppl, sync_dist=True, prog_bar=True)

        self.last_batch_tokens = batch['input_ids'].numel()

        if not hasattr(self, 'total_tokens'):
            self.total_tokens = 0
        self.total_tokens += self.last_batch_tokens
        self.log('total_tokens', self.total_tokens, sync_dist=True, prog_bar=True)

        return lm_loss

    def validation_step(self, batch, batch_idx):
        if self.cfg.experiment.hf_trainer:
            lm_loss, ppl = self.validation_step_hf(batch, batch_idx)
        else:
            input_ids = batch.pop("input_ids")
            lm_logits = self.forward(input_ids).logits

            if "mqar" in self.cfg.task.dataset.class_name.lower():
                labels = batch.pop("label")
                labels = labels.long()
                shift_logits = lm_logits.contiguous()
                labels = labels.contiguous()

            else:
                labels = batch.pop("labels")
                shift_logits = lm_logits[:, :-1, :].contiguous()
                labels = labels[:, 1:].contiguous()

            labels = labels.to(lm_logits.device)

            lm_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            ppl = torch.exp(lm_loss)
            
        self.log("valid_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        self.log("valid_ppl", ppl, sync_dist=True, prog_bar=True)


    def configure_optimizers(self):
        # init optimizer
        if self.cfg.optimizer.optimizer_type.lower() == "adamw":
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
                num_warmup_steps=self.cfg.lr_scheduler.warmup_steps,
                num_training_steps=self.cfg.optimizer.num_training_steps,
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
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


class TokenCountCallback(Callback):
    def __init__(self, max_tokens):
        super().__init__()
        self.max_tokens = max_tokens
        self.total_tokens = 0

    def on_batch_end(self, trainer, pl_module):
        # update token nums
        self.total_tokens += pl_module.last_batch_tokens
        if self.total_tokens >= self.max_tokens:
            trainer.should_stop = True


@hydra.main(config_path='../configs/', config_name='train_config', version_base='1.1')
def main(config):
    print_c(OmegaConf.to_yaml(config))
    pl.seed_everything(config.experiment.seed, workers=True)
    
    model_root_dir = config.platform.hf_model_path
    save_root_dir = config.platform.exp_path
    data_root_dir = config.platform.dataset_path

    # load model and tokenizer
    use_custom_module = False
    if hasattr(config.model, "use_custom_module"):
        use_custom_module = config.model.use_custom_module

    if config.experiment.low_rank_train:
        model, tokenizer = get_low_rank_model_tokenizer(model_root_dir, config.model, use_custom_module=use_custom_module)
    else:
        model, tokenizer = get_model_tokenizer(model_root_dir, config.model, use_custom_module=use_custom_module)
    print_c(model, "magenta")

    # load data
    data_module = CustomDatamodule(config.task, data_root_dir, tokenizer)
    
    # load experiment
    experiment = Experiment(model, config, tokenizer=tokenizer, state="train")

    # init logger
    tb_logger = TensorBoardLogger(
        save_dir=save_root_dir, 
        name=f"{config.exp_task}",
        version=config.mark
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckpt_monitor = ModelCheckpoint(
        save_top_k=config.experiment.save_top_k, 
        dirpath =os.path.join(tb_logger.log_dir, "checkpoints"), 
        monitor=config.experiment.monitor_metric,
        filename=f"{config.model_name}-{config.exp_task}"+"-{epoch:02d}",
        save_last=True,
        mode='min',
        save_weights_only=True, # only s  te dict
        every_n_train_steps=config.experiment.every_n_train_steps,
    )
    token_monitor = TokenCountCallback(max_tokens=10e9)
    # init strategy
    deepspeed_trainer, pl_trainer = None, None
    if config.experiment.use_deepspeed:
        log_c("Using DeepSpeed", "yellow")
        deepspeed_trainer = Trainer(
            default_root_dir=os.path.join(tb_logger.log_dir , "checkpoints"),
            logger=tb_logger,
            callbacks=[lr_monitor, ckpt_monitor, token_monitor],
            check_val_every_n_epoch=1 if data_module.val_dataloader is not None else 1000000,  # set a large number if no validation set
            strategy=DeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_params_device='cpu',
                offload_parameters=True,
                partition_activations=True,
                contiguous_memory_optimization=False,
                cpu_checkpointing=True,
                logging_level=logging.INFO,
                precision_plugin="bf16",
            ),
            precision="bf16",
            accumulate_grad_batches=config.experiment.accumulate_grad_batches,
            enable_checkpointing=True,
            max_steps=config.optimizer.num_training_steps,
            devices=config.experiment.device_num,
            gradient_clip_val=1.0,
            enable_model_summary=True,
            num_sanity_val_steps=5,
            fast_dev_run=5 if config.experiment.debug else False # for debugging
        )
        deepspeed_trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    else:
        log_c("Using pytorch lightning", "yellow")
        pl_trainer = Trainer(
            default_root_dir=os.path.join(tb_logger.log_dir , "checkpoints"),
            logger=tb_logger,
            callbacks=[lr_monitor, ckpt_monitor, token_monitor],
            check_val_every_n_epoch=1 if data_module.val_dataloader is not None else 1000000,  # set a large number if no validation set
            strategy=DDPStrategy(find_unused_parameters=True),
            precision="bf16",
            max_steps=config.optimizer.num_training_steps,
            devices=config.experiment.device_num,
            gradient_clip_val=1,
            enable_model_summary=True,
            num_sanity_val_steps=5,
            accumulate_grad_batches=config.experiment.accumulate_grad_batches,
            fast_dev_run=5 if config.experiment.debug else False # for debugging
        )

    trainer = pl_trainer if pl_trainer is not None else deepspeed_trainer
    trainer.fit(experiment, datamodule=data_module)

if __name__ == '__main__':
    main()








