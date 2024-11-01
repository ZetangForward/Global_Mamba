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
        test_path = "MQAR/mqar-v6-valid.jsonl"

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
        
    
    




    

