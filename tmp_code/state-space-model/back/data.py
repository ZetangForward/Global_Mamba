from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


############################################################
#                       MQAR DATASET                      #
############################################################
VOCAB_SIZE = 8192
    def allocate_num_examples(num_key_pairs):
        """
        分配num_examples给每个key_pair，使得总和为800，并且每个分配是100的整数倍。
        """
        # import pdb;pdb.set_trace()
        allocations = [8//num_key_pairs *100] * num_key_pairs
        total_allocated = sum(allocations)
        i = num_key_pairs - 1  # 从最后一个配置开始分配

        # 从后往前分配，直到总分配量达到800
        while total_allocated < 800:
            allocations[i] += 100  # 每次增加100
            total_allocated += 100
            i -= 1  # 向前移动
            
        return allocations

    # 存储每个配置的num_examples
    num_examples_list = []

    for gap in [2**i for i in range(2, 15)]:
        valid_key_pairs = [key_pair for key_pair in [2, 4, 8, 16, 32, 64, 128, 256] if gap >= key_pair * 2]
    
        # 根据有效的key_pair数量分配num_examples
        allocations = allocate_num_examples(len(valid_key_pairs))
        
        for key_pair, num_examples in zip(valid_key_pairs, allocations):
            input_seq_len = gap + 2 * key_pair
            test_data = MQARDataset.build_gap_dataset(
                        vocab_size=8192,    
                        input_seq_len=input_seq_len,
                        num_kv_pairs=key_pair,
                        num_examples=num_examples,
                        fixed_gap=gap, 
                        insert_out_word=False
                    )
            test_data_path = f"/public/home/ljt/tzc/data/MQAR/fixed_gaps/test_C8192_D{key_pair}_gap{gap}_in.pkl"
            auto_save_data(test_data,test_data_path)
  
  
  
  
   # train_len_num = [512, 1024, 2048, 4096, 8192] 
    # train_kv_num = [32, 64, 128, 256, 512]
    # for i in range(0,len(train_kv_num)):
    #     input_seq_len = train_len_num[i]
    #     number_kv_pairs = train_kv_num[i]
    #     test_data = MQARDataset.build_dataset(
    #         vocab_size=8192, 
    #         input_seq_len=input_seq_len,
    #         num_kv_pairs=number_kv_pairs,
    #         num_examples=100000,
    #         power_a=0.01,
    #         tokenizer=None,
    #         )
    #     data_path = "/public/home/ljt/tzc/data/MQAR/" + "train_C8192_N"+str(input_seq_len) + "_D"+str(number_kv_pairs)+".pkl"
    #     auto_save_data(test_data,data_path)
        
    # for input_seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
    #     for number_kv_pairs in [64]:
    #         try:
    #             test_data = MQARDataset.build_dataset(
    #                 vocab_size=8192, 
    #                 input_seq_len=input_seq_len,
    #                 num_kv_pairs=number_kv_pairs,
    #                 num_examples=3000,
    #                 power_a=0.01,
    #                 tokenizer=None,
    #                 )
    #             # data_path = "/nvme/zecheng/data/MQAR/test_for_test.pkl"
    #             data_path = "/public/home/ljt/tzc/data/MQAR/" + "test_C8192_N"+str(input_seq_len) + "_D"+str(number_kv_pairs)+".pkl"
    #             auto_save_data(test_data,data_path)
    #         except:
    #             print(input_seq_len,number_kv_pairs,"save+-failed")
    # test_data = MQARDataset.build_dataset(
    #             vocab_size=8192, 
    #             input_seq_len=512,
    #             num_kv_pairs=32,
    #             num_examples=3000,
    #             power_a=0.01,
    #             tokenizer=None,
    #             )
    # data_path = "/public/home/ljt/tzc/data/MQAR/test_C8192_N512_D32.pkl"
    # # data_path = "/aifs4su/ziliwang/txw/InternLM/zecheng/data/MQAR/" + "test_C8192_N"+str(input_seq_len) + "_D"+str(number_kv_pairs)+".pkl"
    # auto_save_data(test_data,data_path)
    # all_train_data = []
    # all_test_data = []
    # train_configs = [
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 100000, "num_kv_pairs": 4},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 128, "num_examples": 20000, "num_kv_pairs": 8},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 20000, "num_kv_pairs": 16},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 20000, "num_kv_pairs": 32},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 20000, "num_kv_pairs": 64},
    # ]
    # test_configs = [
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 1000, "num_kv_pairs": 4},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 1000, "num_kv_pairs": 8},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 1000, "num_kv_pairs": 16},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 128, "num_examples": 1000, "num_kv_pairs": 32},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 1000, "num_kv_pairs": 64},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 512, "num_examples": 1000, "num_kv_pairs": 128},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 1024, "num_examples": 1000, "num_kv_pairs": 256},
    # ]

    for train_config in train_configs:
        train_data = MQARDataset.build_dataset(
                    vocab_size=8192, 
                    input_seq_len=train_config["input_seq_len"],
                    num_kv_pairs=train_config["num_kv_pairs"],
                    num_examples=train_config["num_examples"],
                    power_a=0.01,
                    tokenizer=None,
                    insert_out_word=False
                )
        all_train_data.extend(train_data)
    train_data_path = "/nvme1/zecheng/data/MQAR/train_based_in.pkl"
    auto_save_data(all_train_data,train_data_path)
    
    for test_config in test_configs:
        input_seq_len = test_config["input_seq_len"]
        num_kv_pairs = test_config["num_kv_pairs"]
        test_data = MQARDataset.build_dataset(
                    vocab_size=8192,    
                    input_seq_len=input_seq_len,
                    num_kv_pairs=num_kv_pairs,
                    num_examples=test_config["num_examples"],
                    power_a=0.01,
                    tokenizer=None,
                    insert_out_word=False
                )
        # all_test_data.extend(test_data)
        test_data_path = f"/nvme1/zecheng/data/MQAR/test_C8192_N{input_seq_len}_D{num_kv_pairs}_in.pkl"
        auto_save_data(test_data,test_data_path)


class TextFillingDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", full_modeling=True, *args, **kwargs):
        super(TextFillingDataset).__init__()
        self.split = split
        self.content = content
        self.max_text_length = kwargs['max_text_length']
        self.tokenizer = tokenizer
        self.full_modeling = full_modeling
        self.template1 = "Beginning: {s1} {s2} {s3}\nEnding: {s5}\nMiddle: "
        self.template2 = "Beginning: {s1} {s2} {s3}\nEnding: {s5}\nMiddle: {s4}"
        
    def __getitem__(self, index):
        sample = self.content[index]
        s1 = sample["sentence1"]
        s2 = sample["sentence2"]
        s3 = sample["sentence3"]
        s4 = sample["sentence4"]
        s5 = sample["sentence5"]
        
        if not self.full_modeling:
            prompt = self.template1.format(s1=s1, s2=s2, s3=s3, s5=s5)
            
            tokenized_prompt = self.tokenizer(
                prompt,  
                truncation=True, 
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            prompt_ids = tokenized_prompt.input_ids[0]
            label_ids = self.tokenizer(s4, return_tensors="pt").input_ids[0]
            if self.split == "test":
                return {
                    "input_ids": prompt_ids,
                    "labels": label_ids,
                }
            
            prompt_mask = tokenized_prompt.attention_mask[0]
            prompt_sential = torch.empty_like(prompt_ids).fill_(self.tokenizer.pad_token_id)
            
            remain_length = self.max_text_length - prompt_ids.size(0)
            
            tokenized_mid = self.tokenizer(
                s4,  
                truncation=True, 
                padding="max_length",
                max_length=remain_length,
                return_tensors="pt",
            )
            label_ids = tokenized_mid.input_ids[0]
            label_attention_mask = tokenized_prompt.attention_mask[0]
            label_sentinel = label_ids
            
            input_ids = torch.concatenate([prompt_ids, label_ids], dim=0)
            tok_seq = torch.concatenate([prompt_sential, label_sentinel], dim=0)
            attention_mask = torch.concatenate([prompt_mask, label_attention_mask], dim=0)
            
            labels = torch.where(
                tok_seq != self.tokenizer.pad_token_id, tok_seq, -100
            )
        
        else:
            prompt = self.template2.format(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5)
            
            tokenized_prompt = self.tokenizer(
                prompt,  
                truncation=True, 
                padding="max_length",
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            input_ids = tokenized_prompt.input_ids[0]
            attention_mask = tokenized_prompt.attention_mask[0]
            
            labels = torch.where(
                input_ids != self.tokenizer.pad_token_id, input_ids, -100
            )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.content)


class custom_datamodule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.prepare_data_per_node = True
        self.dataset_kwargs = {
            "max_text_length": self.cfg.max_seq_length,
        }
        
    def setup(self, stage: str = 'fit') -> None:
        self.test_dataset = None
        if self.cfg.inference_mode:
            self.test_data = auto_read_data(self.cfg.test_data_path)
            self.test_dataset = TextFillingDataset(
                content=self.test_data, 
                tokenizer=self.tokenizer, 
                full_modeling=False,
                split="test",
                **self.dataset_kwargs,
            )
        else:
            content = auto_read_data(self.cfg.file_path)
            min_valid_num = min(1000, len(content)*0.1)
            self.valid_data = content[:min_valid_num]
            self.train_data = content[min_valid_num:]
            
            self.train_dataset = TextFillingDataset(
                content=self.train_data, 
                tokenizer=self.tokenizer, 
                split="train",
                **self.dataset_kwargs,
            )
            
            self.valid_dataset = TextFillingDataset(
                content=self.valid_data, 
                tokenizer=self.tokenizer, 
                split="valid",
                **self.dataset_kwargs,
            )
            print_c(f"num of train samples: {len(self.train_dataset)}", color='magenta')
            print_c(f"num of valid samples: {len(self.valid_dataset)}", color='magenta')

            
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.train_batch_size, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=True, shuffle=True, 
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.valid_dataset, batch_size=self.cfg.val_batch_size, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
        )
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset, batch_size=1, 
                num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
            )
        return None
    
    
if __name__ == "__main__":
    file_path = "/nvme/zecheng/data/roc_stories/ROCStories_winter2017.csv"
    tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/gpt-neo-1.3B")
    data_module = custom_datamodule(file_path, tokenizer)
    raw_data = data_module.content
    import pdb; pdb.set_trace()