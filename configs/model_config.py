class ModelConfig:
    """ModelConfig class to return model configurations for different models.
    """
    
    def __init__(self, model_name_or_path, tokenizer_name_or_path = None, ckpt_path=None, conv1d_configs=None, *args, **kwargs):

        self.cfg = self.return_config(model_name_or_path, tokenizer_name_or_path, ckpt_path, conv1d_configs, *args, **kwargs)

    def return_config(self, model_name_or_path, tokenizer_name_or_path, ckpt_path=None, conv1d_configs=None, *args, **kwargs):
        if tokenizer_name_or_path is None: tokenizer_name_or_path = model_name_or_path

        if "mamba" in model_name_or_path.lower(): # SSM model config
          
            if conv1d_configs is None:
                conv1d_configs = {}
            conv1d_configs.update(kwargs)
            

            if "130" in model_name_or_path.lower():  # 130M mamba
                return self.mamba_config(model_name_or_path='mamba-130m-hf', ckpt_path=ckpt_path, 
                                        load_model_state_dict=ckpt_path is not None, conv1d_configs=conv1d_configs, 
                                         *args, **kwargs) 
            if "370" in model_name_or_path.lower():  # 370M mamba
                return self.mamba_config(model_name_or_path='mamba-370m-hf', ckpt_path=ckpt_path, 
                                        load_model_state_dict=ckpt_path is not None, conv1d_configs=conv1d_configs, 
                                         *args, **kwargs) 
            elif "1_4b" in model_name_or_path.lower(): # 1.4b mamba
                return self.mamba_config(model_name_or_path="mamba-1.4b-hf", ckpt_path = ckpt_path, 
                                        conv1d_configs=conv1d_configs, load_model_state_dict=ckpt_path is not None, *args, **kwargs)
            elif "tiny" in model_name_or_path.lower(): # for tiny model (2 layers)
                # import re
                # exp_name = kwargs["exp_name"]
                ssm_size = 16
                return self.tiny_mamba_config(model_name_or_path="mamba-370m-hf", tokenizer_name_or_path="mamba-370m-hf", 
                                             ckpt_path=ckpt_path, load_model_state_dict=ckpt_path is not None, 
                                             conv1d_configs=conv1d_configs, vocab_size=20480, ssm_size=ssm_size, *args, **kwargs)
        
        elif "deepseek" in model_name_or_path.lower():
            return self.deepseek_config(model_name_or_path="deepseek-coder-1.3b-base", ckpt_path=ckpt_path, 
                                        load_model_state_dict=ckpt_path is not None)
        
        elif "metatransformer" in model_name_or_path.lower():
            return self.metatransformer_config(model_name_or_path="metatransformer", ckpt_path=ckpt_path, 
                                               load_model_state_dict=ckpt_path is not None, 
                                               conv1d_configs={"kernel_sizes": 4, "token_mixer_type": "vanilla_conv1d"})
        
        elif "long_gpt_neo" in model_name_or_path.lower():  #s long_pythia
            return self.gpt_neo_config(model_name_or_path="gpt-neo-1.3B", ckpt_path=ckpt_path, 
                                      load_model_state_dict=ckpt_path is not None, use_custom_module=True)
        
        elif "tinyllama" in model_name_or_path.lower():  # tinyllama
            return self.tinyllama_config(model_name_or_path="TinyLlama-1.1B-intermediate-step-1431k-3T", ckpt_path=ckpt_path, load_model_state_dict=ckpt_path is not None)

        elif "gpt-neo" in model_name_or_path.lower():
            return self.gpt_neo_config(model_name_or_path="gpt-neo-125m", 
                                       tokenizer_name_or_path="mamba-370m-hf",
                                       ckpt_path=ckpt_path, load_model_state_dict=ckpt_path is not None)
        
        elif "pythia" in model_name_or_path.lower():
            return self.pythia_config(model_name_or_path="pythia-160m", 
                                       tokenizer_name_or_path="pythia-160m",
                                       ckpt_path=ckpt_path, load_model_state_dict=ckpt_path is not None)
        
        elif "gla" in model_name_or_path.lower():
            return self.gla_config(model_name_or_path="gla-1.3b", ckpt_path=ckpt_path, load_model_state_dict=ckpt_path is not None)
        
        elif "rwkv" in model_name_or_path.lower():
            return self.rwkv_config(model_name_or_path="rwkv", ckpt_path=ckpt_path, load_model_state_dict=ckpt_path is not None)

        elif "hgrn" in model_name_or_path.lower():
            return self.rwkv_config(model_name_or_path="hgrn", ckpt_path=ckpt_path, load_model_state_dict=ckpt_path is not None)
    
        elif "hyena" in model_name_or_path.lower():
            return self.hyena_config(model_name_or_path="hyena", ckpt_path=ckpt_path, load_model_state_dict=ckpt_path is not None)

        elif "based" in model_name_or_path.lower():
            return self.based_config(model_name_or_path="based", ckpt_path=ckpt_path, load_model_state_dict=ckpt_path is not None)


    def tiny_mamba_config(self, model_name_or_path=None, tokenizer_name_or_path=None, ckpt_path=None, load_model_state_dict = False,
                          conv1d_configs = None, use_custom_module = False, tiny_mamba_configs = None, vocab_size = 20480, ssm_size=16,  **kwargs):  
        import math
        tiny_mamba_configs = {"num_hidden_layers": 8, "hidden_size": 256, "intermediate_size": 512, 
                              "time_step_rank": 32, "vocab_size": vocab_size, "ssm_state_size": ssm_size}
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": tokenizer_name_or_path, 
                "ckpt_path": ckpt_path, "load_model_state_dict": load_model_state_dict, "conv1d_configs": conv1d_configs, 
                "use_custom_module": use_custom_module, "tiny_mamba_config": tiny_mamba_configs, **kwargs}
    

    def mamba_config(self, model_name_or_path, ckpt_path=None, load_model_state_dict=False, conv1d_configs=None,exp_name=None,**kwargs):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": model_name_or_path, "ckpt_path": ckpt_path,
                "load_model_state_dict": load_model_state_dict, "conv1d_configs": conv1d_configs,  **kwargs}
    

    def deepseek_config(self, model_name_or_path, ckpt_path, load_model_state_dict):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": model_name_or_path,
                "load_model_state_dict": load_model_state_dict, "ckpt_path": ckpt_path}

    def metatransformer_config(self, model_name_or_path, ckpt_path, load_model_state_dict, conv1d_configs):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": "mamba-370m-hf", "ckpt_path": ckpt_path, 
                "load_model_state_dict": load_model_state_dict, "conv1d_configs": conv1d_configs, "use_custom_module": False}

    def gpt_neo_config(self, model_name_or_path,tokenizer_name_or_path ,ckpt_path, load_model_state_dict, use_custom_module=False):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": tokenizer_name_or_path,
                "load_model_state_dict": load_model_state_dict, "ckpt_path": ckpt_path, "use_custom_module": use_custom_module}
    
    def tinyllama_config(self, model_name_or_path, ckpt_path, load_model_state_dict, use_custom_module=False):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": model_name_or_path,
                "load_model_state_dict": load_model_state_dict, "ckpt_path": ckpt_path, "use_custom_module": use_custom_module}
    
    def pythia_config(self, model_name_or_path,tokenizer_name_or_path ,ckpt_path, load_model_state_dict, use_custom_module=False):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": tokenizer_name_or_path,
                "load_model_state_dict": load_model_state_dict, "ckpt_path": ckpt_path, "use_custom_module": use_custom_module}

    def gla_config(self, model_name_or_path ,ckpt_path, load_model_state_dict, use_custom_module=False):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": "/nvme/hf_models/pythia-160m",
                "load_model_state_dict": load_model_state_dict, "ckpt_path": ckpt_path, "use_custom_module": use_custom_module}
    
    def rwkv_config(self, model_name_or_path ,ckpt_path, load_model_state_dict, use_custom_module=False):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": "/nvme/hf_models/pythia-160m",
                "load_model_state_dict": load_model_state_dict, "ckpt_path": ckpt_path, "use_custom_module": use_custom_module}

    def HGRN_config(self, model_name_or_path ,ckpt_path, load_model_state_dict, use_custom_module=False):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": "/nvme/hf_models/pythia-160m",
                "load_model_state_dict": load_model_state_dict, "ckpt_path": ckpt_path, "use_custom_module": use_custom_module}
    
    def hyena_config(self, model_name_or_path ,ckpt_path, load_model_state_dict, use_custom_module=False):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": "/nvme/hf_models/pythia-160m",
                "load_model_state_dict": load_model_state_dict, "ckpt_path": ckpt_path, "use_custom_module": use_custom_module}
    
    def based_config(self, model_name_or_path ,ckpt_path, load_model_state_dict, use_custom_module=False):
        return {"model_name_or_path": model_name_or_path, "tokenizer_name_or_path": "/nvme/hf_models/pythia-160m",
                "load_model_state_dict": load_model_state_dict, "ckpt_path": ckpt_path, "use_custom_module": use_custom_module}
    
    