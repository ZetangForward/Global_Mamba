import os
import sys
sys.path.append(os.getcwd()) 
import torch
import hydra
import transformers
import pytorch_lightning as pl
from transformers import AutoTokenizer, GPTNeoForCausalLM
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from custom_dataset.data import custom_datamodule
from modelzipper.tutils import *
from torch import optim, Tensor 


class Experiment(pl.LightningModule):
    
    def __init__(self, model, config, tokenizer=None, state="train") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.train()
        self.cfg = config
        self.exp_cfg = config.experiment
        self.tokenizer = tokenizer
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(**input, **kwargs)

    def training_step(self, batch, batch_idx):
        lm_loss = self.forward(batch).loss
        self.log("train_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        return lm_loss

    def validation_step(self, batch, batch_idx):
        lm_loss = self.forward(batch).loss
        self.log("val_lm_loss", lm_loss, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = transformers.AdamW(self.model.parameters(), lr=self.exp_cfg.lr)

        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, self.exp_cfg.warmup_steps, self.exp_cfg.num_training_steps)

        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'custom_scheduler',
            'interval': 'step',  # Ensure learning rate updates per step
            'frequency': 1,  # Optional: If you want to make sure it updates every step
        }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


@hydra.main(config_path='../../configs', config_name='train_gpt', version_base='1.1')
def main(config):
    pl.seed_everything(config.experiment.seed, workers=True)
    
    # load model and tokenizer
    model, tokenizer = get_model_tokenizer(config.model, config.tokenizer)
    
    # load data
    data_module = custom_datamodule(config.dataset, tokenizer)
    
    # load experiment
    experiment = Experiment(model, config, tokenizer=tokenizer, state="train")
    
    # init logger
    tb_logger = TensorBoardLogger(
        save_dir=config.experiment.model_save_dir, 
        name=f"{config.experiment.task}",
        version=config.experiment.version
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        # accelerator=config.device.accelerator,
        precision=config.device.precision,
        devices=config.device.device_num,
        default_root_dir=os.path.join(tb_logger.log_dir , "checkpoints"),
        logger=tb_logger,
        callbacks=[
            lr_monitor,
            ModelCheckpoint(
                save_top_k=5, 
                dirpath =os.path.join(tb_logger.log_dir, "checkpoints"), 
                monitor="val_lm_loss",
                filename=f"gpt-noe-{config.experiment.task}"+"-{epoch:02d}",
                save_last=True,
                mode='min',
            ),
        ],
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_steps=config.experiment.num_training_steps,
        gradient_clip_val=1,
        enable_model_summary=True,
        num_sanity_val_steps=20,
        # fast_dev_run=5 # for debugging
    )

    trainer.fit(experiment, datamodule=data_module)

def get_model_tokenizer(model_config, tokenizer_config):
    model = GPTNeoForCausalLM.from_pretrained(model_config.model_name_or_path, torch_dtype=torch.bfloat16).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_name_or_path)
    if "gpt-neo" in tokenizer_config.tokenizer_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


if __name__ == '__main__':
    main()








