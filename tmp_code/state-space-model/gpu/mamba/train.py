
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
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from modelzipper.tutils import *
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from utils import get_model_tokenizer, CustomDatamodule


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
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        lm_loss = outputs.loss
        ppl = torch.exp(lm_loss)
        self.log("train_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        self.log("train_ppl", ppl, sync_dist=True, prog_bar=True)
        return lm_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        lm_loss = outputs.loss
        ppl = torch.exp(lm_loss)
        
        self.log("valid_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        self.log("valid_ppl", ppl, sync_dist=True, prog_bar=True)
      

    def training_step(self, batch, batch_idx):
        if self.cfg.experiment.hf_trainer:
            return self.training_step_hf(batch, batch_idx)

        else:
            input_ids = batch.pop("input_ids")
            lm_logits = self.forward(input_ids).logits
            # import pdb;pdb.set_trace()
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
            
            self.log("train_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
            self.log("train_ppl", ppl, sync_dist=True, prog_bar=True)
            return lm_loss

    def validation_step(self, batch, batch_idx):
        if self.cfg.experiment.hf_trainer:
            self.validation_step_hf(batch, batch_idx)
        else:
            input_ids = batch.pop("input_ids")
            lm_logits = self.forward(input_ids).logits
            # import pdb;pdb.set_trace()
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
                num_training_steps=self.cfg.experiment.num_training_steps,
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
                self.cfg.experiment.num_training_steps, 
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
    

@hydra.main(config_path='../configs/', config_name='train_config', version_base='1.1')
def main(config):

    # print_c(f"Conduct Experiment: {config.exp_task} | Model: {config.model} | State: {config.state} | Platform: {config.platform}", "magenta")
    print_c(OmegaConf.to_yaml(config), "yellow")
    
    model_root_dir = config.platform.hf_model_path
    save_root_dir = config.platform.exp_path
    data_root_dir = config.platform.dataset_path

    pl.seed_everything(config.experiment.seed, workers=True)
    
    # load model and tokenizer
    use_custom_module = False
    if hasattr(config.model, "use_custom_module"):
        use_custom_module = True

    if not config.experiment.low_rank_train:
        model, tokenizer = get_model_tokenizer(model_root_dir, config.model, use_custom_module=use_custom_module)
    else:
        model, tokenizer = get_low_rank_model_tokenizer(model_root_dir, config.model, use_custom_module=use_custom_module)
        
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
        filename=f"mamba-{config.exp_task}"+"-{epoch:02d}",
        save_last=True,
        mode='min',
        save_weights_only=True, # only save state dict
    )
    
    # # TODO: add deepspeed strategy
    # deepspeed_config = {
    #     "zero_allow_untested_optimizer": True,
    #     "zero_optimization": {
    #         "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
    #         "offload_optimizer": {"device": "cpu"},  # Enable Offloading optimizer state/calculation to the host CPU
    #         "contiguous_gradients": True,  # Reduce gradient fragmentation.
    #         "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
    #         "allgather_bucket_size": 2e8,  # Number of elements to all gather at once.
    #         "reduce_bucket_size": 2e8,  # Number of elements we reduce/allreduce at once.
    #     },
    # }

    
    # strategy = DeepSpeedStrategy(accelerator='gpu', config=deepspeed_config)
    deepspeed_trainer, pl_trainer = None, None
    if config.experiment.use_deepspeed:
        deepspeed_trainer = Trainer(
            default_root_dir=os.path.join(tb_logger.log_dir , "checkpoints"),
            logger=tb_logger,
            callbacks=[lr_monitor, ckpt_monitor],
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
                precision_plugin="bf16-mixed",
            ),
            accumulate_grad_batches=8,
            enable_checkpointing=True,
            max_steps=config.experiment.num_training_steps,
            devices=config.experiment.device_num,
            gradient_clip_val=1,
            enable_model_summary=True,
            num_sanity_val_steps=5,
            fast_dev_run=5 if config.experiment.debug else False # for debugging
        )
        deepspeed_trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    else:
        pl_trainer = Trainer(
            default_root_dir=os.path.join(tb_logger.log_dir , "checkpoints"),
            logger=tb_logger,
            callbacks=[lr_monitor, ckpt_monitor],
            check_val_every_n_epoch=1 if data_module.val_dataloader is not None else 1000000,  # set a large number if no validation set
            strategy=DDPStrategy(find_unused_parameters=True),
            # strategy="deepspeed_stage_2_offload",
            precision="bf16-mixed",
            max_steps=config.experiment.num_training_steps,
            devices=config.experiment.device_num,
            gradient_clip_val=1,
            enable_model_summary=True,
            num_sanity_val_steps=5,
            fast_dev_run=5 if config.experiment.debug else False # for debugging
        )

    trainer = pl_trainer if pl_trainer is not None else deepspeed_trainer
    
    trainer.fit(experiment, datamodule=data_module)

if __name__ == '__main__':
    main()








