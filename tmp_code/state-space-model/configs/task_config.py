from modelzipper.tutils import *

############################################################
##              ALL DATA CONFIG SHOULD CONCLUDE:          ## 
## - split / data_name / data_path / processed_data_path  ##
## - module / dataset_class_name / collate_fn_name        ##
## - inference_mode / max_seq_length / cluster_batch      ##
## - batch_size / nworkers / pin_memory                   ##
############################################################

class TaskConfig:
    def __init__(self, data_name, data_path=None, processed_data_path=None, module=None, class_name=None, nworkers=0, 
                 max_seq_length=4096, train_batch_size=1, val_batch_size=1, inference_mode=False, pin_memory=False, 
                 cluster_batch=False, data_type="custom", data_dir=None, **other_cfgs):
        # import pdb;pdb.set_trace()
        self.cfg = self.return_config(data_name, processed_data_path, train_batch_size, val_batch_size, 
                                      inference_mode, max_seq_length, nworkers, data_type, pin_memory, data_dir, **other_cfgs)

    def return_config(self, data_name, processed_data_path, train_batch_size, val_batch_size, 
                      inference_mode, max_seq_length, nworkers, data_type, pin_memory, data_dir, *args, **kwargs):

        if "mqar" in data_name.lower(): 
            # import pdb;pdb.set_trace()
            return self.mqar_config(train_batch_size=train_batch_size if not inference_mode else None, max_seq_length=max_seq_length,
                                    val_batch_size=val_batch_size, pin_memory=pin_memory, nworkers=nworkers, data_dir=data_dir, **kwargs)
        elif "pg19" in data_name.lower():
            return self.pg19_config(max_seq_length)
        elif "copy" in data_name.lower(): 
            return self.copy_config(inference_mode,train_batch_size, val_batch_size, max_seq_length, processed_data_path,)
        elif "longbench" in data_name.lower(): 
            return self.longbench_config(nworkers, val_batch_size)
        elif "passkey" in data_name.lower(): 
            return self.passkey_config(max_seq_length)
        elif "longalpaca" in data_name.lower(): 
            return self.longalpaca_config(max_seq_length, train_batch_size, val_batch_size, nworkers)
        elif "slimpajama" in data_name.lower(): 
            return self.slimpajama_config(max_seq_length, train_batch_size, val_batch_size, nworkers)


    #  MQAR/mqar-v6-k4v8-2k-withkeyloss-train.jsonl
    def mqar_config(self, train_batch_size, val_batch_size, nworkers, pin_memory, max_seq_length=1024, data_dir=None, **kwargs):

        train_path = "MQAR/mqar-v0-standard-512-train.jsonl"
        valid_path = "MQAR/mqar-v0-standard-2k-valid.jsonl"
        test_path = "MQAR/mqar-v6-k4v8-8k-valid.jsonl"

        if data_dir is not None:
            train_path = os.path.join(data_dir, "train.jsonl")
            valid_path = os.path.join(data_dir, "valid.jsonl")
            test_path = os.path.join(data_dir,"test.jsonl")
        if kwargs and kwargs.get("train_path"):
            train_path = kwargs.get("train_path")
        if kwargs and kwargs.get("valid_path"):
            valid_path = kwargs.get("valid_path")   
        if kwargs and kwargs.get("test_path"):
            test_path = kwargs.get("test_path")

        return {"task_name": "MQAR", "batch_tokens": None, "train_batch_size": train_batch_size, 
                "dataset": [
                    {"split": "train", "data_name": "MQAR", "data_path": None, "processed_data_path": train_path, 
                    "module": 'custom_dataset.mqar', "dataset_class_name": 'MQARDataset', "collate_fn_name": "mqar_collate_fn", "max_seq_length": max_seq_length,
                    "nworkers": nworkers, "batch_size": train_batch_size, "inference_mode": False, "pin_memory": pin_memory, "cluster_batch": True},
                    {"split": "valid", "data_name": "MQAR", "data_path": None, "processed_data_path": valid_path, 
                    "module": 'custom_dataset.mqar', "dataset_class_name": 'MQARDataset', "collate_fn_name": "mqar_collate_fn", "max_seq_length": max_seq_length,
                    "nworkers": nworkers, "batch_size": val_batch_size, "inference_mode": False, "pin_memory": pin_memory, "cluster_batch": True},    
                    {"split": "test", "data_name": "MQAR", "data_path": None, "processed_data_path": test_path, 
                    "module": 'custom_dataset.mqar', "dataset_class_name": 'MQARDataset', "collate_fn_name": "mqar_collate_fn", "max_seq_length": None,
                    "nworkers": nworkers, "batch_size": val_batch_size, "inference_mode": True, "pin_memory": pin_memory, "cluster_batch": False},   
                ],
                "other_cfgs": {"max_seq_length": max_seq_length}}
        
    @classmethod
    def copy_config(
        cls, inference_mode,
        train_batch_size, val_batch_size, max_seq_length,
        processed_data_path,
    ):
        if processed_data_path is None:
            processed_data_path="Copy/train.pkl"
        copy_config = {
            "task_name": "Copy",
            'dataset': {
                "data_name": "Copy",
                "data_path": None, 
                "processed_data_path": processed_data_path,
                "module": 'custom_dataset.Copy_ywj', 
                "class_name": 'CopyDataset',
                "nworkers": 4,
                "max_seq_length": max_seq_length,
                "train_batch_size": train_batch_size,
                "val_batch_size": val_batch_size,
                "inference_mode": inference_mode,
                "pin_memory": False,
                "cluster_batch": False,
                "vocab_size": 8192,
            },
            "other_cfgs": {
                "max_generation_length": 48,
                "testing_max_ctx": 128000,
            },
        }
        return copy_config


    def passkey_config(self, max_seq_length):
        return {
            "task_name": "passkey_search", 
            "dataset": {
                "data_name": "PasskeySearch",
                "data_path": "needle/PaulGrahamEssays/*.txt",
                "processed_data_path": "passkey_search/processed_data/128k_500_insert_ids.pkl",
                "module": 'custom_dataset.passkey_search',
                "class_name": 'PasskeySearchDataset',
                "nworkers": 4,
                "max_seq_length": max_seq_length,
                "val_batch_size": 1,
                "inference_mode": True,
                "pin_memory": False,
                "cluster_batch": False,
                "depth": 0.5,
                "key": "The best thing to do in San Francisco is",
                "value": "eat a sandwich and sit in Dolores Park on a sunny day.",
            },
            "other_cfgs": {
                "max_generation_length": 48,
                "testing_max_ctx": max_seq_length,
            },
            "inference_cfg": {
                "save_keys": ['depth', 'ctx_length', 'real_length']
            }
        }
        return passkey_config
    
    
    def longalpaca_config(self, max_seq_length, train_batch_size, val_batch_size, nworkers):
        longalpaca_config = {
            "task_name": "longalpaca", 
            "dataset": [{"split": "train", "data_path": "LongAlpaca-12k/LongAlpaca-12k.json",
                         "processed_data_path": None, "max_seq_length": max_seq_length, "module": 'custom_dataset.longlora',
                         "dataset_class_name": 'LongLoRA', "nworkers": nworkers, "type": "jsonl",
                         "batch_size": train_batch_size, "pin_memory": False, "inference_mode": False,
                         "cluster_batch": True, "collate_fn_name": "custom_collate_fn"}
                        ],
            "other_cfgs": None,
            "batch_tokens": max_seq_length * train_batch_size,
            "train_batch_size": train_batch_size,
        }
        return longalpaca_config
    
    
    def slimpajama_config(self, max_seq_length, train_batch_size, val_batch_size, nworkers):
        slimpajama_config = {
            "task_name": "slimpajama", 
            "dataset": [
                {
                    "split": "train",
                    "data_name": "slim_pajama",
                    "data_path": "/mnt/petrelfs/tangzecheng/local_data/slimpajama-processed/processed_data_2048",
                    "module": 'custom_dataset.slimpajama',
                    "processed_data_path": None,
                    "dataset_class_name": 'Slimpajama',
                    "max_seq_length": max_seq_length,
                    "nworkers": nworkers,
                    "type": "hf",
                    "batch_size": train_batch_size,
                    "pin_memory": True,
                    "inference_mode": False,
                    "cluster_batch": False,
                    "require_process": False,
                },
                {
                    "split": "validation",
                    "data_name": "slim_pajama",
                    "data_path": "/mnt/petrelfs/tangzecheng/local_data/slimpajama-processed/processed_data_2048",
                    "module": 'custom_dataset.slimpajama',
                    "processed_data_path": None,
                    "dataset_class_name": 'Slimpajama',
                    "max_seq_length": max_seq_length,
                    "nworkers": nworkers,
                    "type": "hf",
                    "batch_size": val_batch_size,
                    "pin_memory": True,
                    "inference_mode": False,
                    "cluster_batch": False,
                    "require_process": False,
                },
                {
                    "split": "test",
                    "data_name": "slim_pajama",
                    "data_path": "/mnt/petrelfs/tangzecheng/local_data/slimpajama-processed/processed_data_20488",
                    "module": 'custom_dataset.slimpajama',
                    "processed_data_path": None,
                    "dataset_class_name": 'Slimpajama',
                    "max_seq_length": max_seq_length,
                    "nworkers": nworkers,
                    "type": "hf",
                    "batch_size": val_batch_size,
                    "pin_memory": True,
                    "inference_mode": True,
                    "cluster_batch": False,
                    "require_process": False,
                }
            ],
             "other_cfgs": {
                "max_generation_length": 48,
                "testing_max_ctx": max_seq_length,
            },
            "batch_tokens": max_seq_length * train_batch_size,
            "train_batch_size": train_batch_size  # for logging
        }
        return slimpajama_config

    def longbench_config(self, nworkers, batch_size):
        longbench_config = {
            "task_name": "longbench", 
            "dataset": [{"split": "test", "data_name": "longbench", "data_path": None, "processed_data_path": "/nvme1/zecheng/data/longbench/longbench_all.jsonl",
                         "module": 'custom_dataset.longbench', "dataset_class_name": 'LongBenchDataset', "collate_fn_name": None, "max_seq_length": 64000,
                         "nworkers": nworkers, "pin_memory": False, "inference_mode": True, "cluster_batch": False, "batch_size": batch_size}],
            "other_cfgs": None,
        }
        return longbench_config

    def pg19_config(self, max_seq_length):
        pg19_config = {
            "task_name": "pg19", 
            "dataset": [
                {
                    "split": "test",
                    "data_name": "pg19",
                    "data_path": "/nvme1/zecheng/DeciMamba/hf_cache/data",
                    "module": 'custom_dataset.pg19',
                    "processed_data_path": None,
                    "dataset_class_name": 'PG19Data',
                    "max_seq_length": None,
                    "nworkers": 1,
                    "type": "hf",
                    "batch_size": 1,
                    "pin_memory": True,
                    "inference_mode": True,
                    "cluster_batch": False,
                    "require_process": False,
                },
        
            ],
             "other_cfgs": {
                "max_generation_length": 48,
                "testing_max_ctx": max_seq_length,
            },
            "batch_tokens": max_seq_length * 1,
            "train_batch_size": 1  # for logging
        }
        return pg19_config

    

############################################################
##              ALL DATA CONFIG SHOULD CONCLUDE:          ## 
## - split / data_name / data_path / processed_data_path  ##
## - module / dataset_class_name / collate_fn_name        ##
## - inference_mode / max_seq_length / cluster_batch      ##
## - batch_size / nworkers / pin_memory                   ##
############################################################
    
            




    

