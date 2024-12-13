class OptimizerConfig:
    def __init__(self, opt_name, train_step, warmup_step, learning_rate) -> None:
        self.opt_name = opt_name
        self.train_step = train_step
        self.warmup_step = warmup_step
        
        self.cfg = self.return_config(opt_name, train_step, warmup_step, learning_rate)

    def return_config(self, opt_name, train_step=20000, warmup_step=2000, learning_rate=5e-5):
        if "adawm" in opt_name.lower():   
            return self.adamw_config(train_step, warmup_step, learning_rate) 
        else:
            ...


    def adamw_config(self, num_training_steps, warmup_steps, learning_rate):
        adamw_config = {
            "optimizer_type": "adamw",
            "lr": learning_rate,
            "beta_1": 0.9,
            "beta_2": 0.95,
            "num_training_steps": num_training_steps,
            "warmup_steps": warmup_steps,
            "peak_lr": 0.0002,
            "last_lr": 0.00001,
        }

        return adamw_config


class LR_Scheduler_Config:
    def __init__(self, lr_scheduler_name, train_step, warmup_step) -> None:
        self.lr_scheduler_config = lr_scheduler_name
        self.train_step = train_step
        self.warmup_step = warmup_step
        
        self.cfg = self.return_config(lr_scheduler_name, train_step, warmup_step)

    def return_config(self, lr_scheduler_name, train_step=20000, warmup_step=2000):
        if "cosine" in lr_scheduler_name.lower():   
            return self.consine_schedule_config(train_step, warmup_step) 
        else:
            ...

    def consine_schedule_config(self, num_training_steps, warmup_steps):
        adamw_config = {
            "scheduler_type": "get_cosine_schedule_with_warmup",
            "num_training_steps": num_training_steps,
            "warmup_steps": warmup_steps,
        }

        return adamw_config