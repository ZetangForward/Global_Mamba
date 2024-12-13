from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import random
import numpy as np
import torch
import glob



class CopyDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs["max_seq_length"] 
        self.cluster_batch = kwargs["cluster_batch"]

    @classmethod
    def build_dataset(cls, split, vocab_size, input_seq_len, num_examples, copy_token=4099, tokenizer=None, random_seed=42):
        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if split == 'train':
            # 生成随机序列
            data = torch.randint(2,  vocab_size+2, (num_examples, input_seq_len))

            # 构造复制令牌ID的张量
            copy_token_ids = torch.full((num_examples, 1), copy_token, dtype=torch.long)

            # 构造输入：原始序列后跟复制令牌和原始序列的重复
            inputs = torch.cat((data, copy_token_ids, data), dim=1)

            # 构建标签：标签前半部分全部为-100（忽略索引），从复制令牌开始为原始序列
            labels = torch.full((num_examples, input_seq_len + 1), -100, dtype=torch.long)
            labels = torch.cat((labels, data), dim=1)  # 添加原始序列作为正确的输出
            
            
            
        else:
            # import pdb;pdb.set_trace()
            # 生成随机序列
            data = torch.randint(2,  vocab_size+2, (num_examples, input_seq_len))

            # 构造复制令牌ID的张量
            copy_token_ids = torch.full((num_examples, 1), copy_token, dtype=torch.long)

            # 构造输入：原始序列后跟复制令牌和原始序列的重复
            inputs = torch.cat((data, copy_token_ids), dim=1)

            labels = data.clone()
            
            
        # 构建最终的数据集
        all_data = []
        for i in range(inputs.size(0)):  
            input_list = inputs[i].to(torch.int32)
            label_list = labels[i].to(torch.int32)
            data_dict = {'input': input_list, 'label': label_list, 'input_seq_len': input_seq_len}
            
            all_data.append(data_dict)

        return all_data
        
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        item = self.content[index]
        input_ids = item.pop('input')
        label = item.pop('label')
        # attention_mask = torch.ones(input_ids.shape,dtype=input_ids.dtype)

        # if len(input_ids) < self.max_seq_length:
        #     input_ids = input_ids.tolist()
        #     label = label.tolist()
        #     # Create a padding mask
        #     attention_mask = [1] * len(input_ids) + [0] * (self.max_seq_length - len(input_ids))
        #     # Pad the sequence
        #     input_ids = torch.nn.functional.pad(input=torch.tensor(input_ids), pad=(0, self.max_seq_length - len(input_ids)), mode='constant', value=self.tokenizer.pad_token_id)
        #     label = torch.nn.functional.pad(input=torch.tensor(label), pad=(0, self.max_seq_length - len(label)), mode='constant', value=-100)
        # if len(input_ids) > self.max_seq_length:
            
        attention_mask = torch.ones(input_ids.shape,dtype=input_ids.dtype)
        
        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels" : label
        }

        res.update(item)

        return res

if __name__ == '__main__':
    # tmp = CopyDataset.build_dataset(8192, 4, 20000, 50000,None)

    all_train_data = []
    all_test_data = []
    VOCAB_SIZE = 4096
    train_configs = [
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 20000},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 128, "num_examples": 20000},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 20000},
    ]

    # test_configs = [
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": i, "num_examples": 100}
    #     for i in range(4,512,4)
    # ]

    for train_config in train_configs:
        input_seq_len = train_config["input_seq_len"]
        train_data = CopyDataset.build_dataset(
                    split='train',
                    vocab_size=VOCAB_SIZE, 
                    input_seq_len=input_seq_len,
                    num_examples=train_config["num_examples"],
                    tokenizer=None,
                )

        all_train_data.extend(train_data)
    train_data_path = f"/public/home/ljt/tzc/data/Copy/V4096_train.pkl"
    auto_save_data(all_train_data,train_data_path)
    
    # for test_config in test_configs:
    #     input_seq_len = test_config["input_seq_len"]
    #     test_data = CopyDataset.build_dataset(
    #                 split='test',
    #                 vocab_size=VOCAB_SIZE, 
    #                 input_seq_len=input_seq_len,
    #                 num_examples=test_config["num_examples"],
    #                 tokenizer=None,
    #             )
    #     # all_test_data.extend(test_data)
    #     test_data_path = f"/public/home/ljt/tzc/data/Copy/V4096_N100_test/test_V{VOCAB_SIZE}_L{input_seq_len}.pkl"
    #     auto_save_data(test_data,test_data_path)
        
    # # test_data = CopyDataset.build_dataset(
    # #                     split='test',
    # #                     vocab_size=8192, 
    # #                     input_seq_len=10,
    # #                     num_examples=100,
    # #                     tokenizer=None,
    # #                     copy_token=20000,
    # #                 )
    # #         # all_test_data.extend(test_data)
    # # test_data_path = f"/public/home/ljt/tzc/data/Copy/test_V{VOCAB_SIZE}_{input_seq_len}.pkl"
    # # auto_save_data(test_data,test_data_path)
