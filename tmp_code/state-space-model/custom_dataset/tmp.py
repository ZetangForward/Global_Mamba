from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import random
import numpy as np
import torch
import glob



class MQARDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs["max_seq_length"]
        self.cluster_batch = kwargs["cluster_batch"]
        # import pdb;pdb.set_trace()

    @classmethod
    def build_dataset(cls, vocab_size, num_examples, input_seq_len, num_kv_pairs, power_a, tokenizer, insert_out_word=True, random_non_queries=True , random_seed=42):
        
        assert input_seq_len % 2 == 0, "input_seq_len must be even"
        # assert vocab_size > input_seq_len
        assert num_kv_pairs * 4 <= input_seq_len
        seed = random_seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        context_size = num_kv_pairs * 2
        # import pdb;pdb.set_trace()
        # create keys so that each key is present exactly once in each example
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)

        keys_unshuffled = np.tile(key_choices, (num_examples, 1))
        keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

        values_unshuffled = np.tile(value_choices, (num_examples, 1))
        values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

        # create sequences
        kvs = np.zeros((num_examples, context_size), dtype=np.int64)
        kvs[:, 0::2] = keys
        kvs[:, 1::2] = values

        # compute power law
        space = (input_seq_len - context_size) // 2
        p = power_a * np.arange(1, space + 1) ** (power_a-1)    # 幂律分布
        p = p / p.sum()

        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
        
        # queries and answers
        queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
        np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
        examples = np.concatenate([
            kvs, 
            queries
        ], axis=1)

        labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
        np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

        inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
        # replace all the 0 with random values
        if random_non_queries:                          # 随机插入非 key-value值
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size+1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

        all_test_data = []

        for i in range(inputs.size(0)):  
            input_list = inputs[i].to(torch.int32)
            # label_idx = torch.nonzero(labels[i] != -100).flatten().to(torch.int32)
            # label_value = torch.index_select(labels[i], 0, label_idx).to(torch.int32)
            
            # data_dict = {'input': input_list, 'label_idx': label_idx, 'label_value': label_value}
            label_list = labels[i].to(torch.int32)
            # import pdb;pdb.set_trace()
            meta_data = []
            target_idx = torch.where(label_list!=-100)[0]
            target_value = label_list[target_idx]
            key_value = input_list[target_idx]
            for idx in range(len(key_value)):
                key = key_value[idx]
                key_idx = torch.where(input_list==key)[0][0]
                meta_data.append((int(key_idx), int(target_idx[idx]),{int(key): int(target_value[idx])}))
            
            data_dict = {'input': input_list, 'label': label_list, 'meta_data': meta_data, 'ctx_length': int(input_seq_len), 'num_kv_pairs': int(num_kv_pairs) }
            # data_dict = {'ctx_length': input_seq_len, 'V_id': (torch.nonzero(label_list != -100).squeeze() + 1).tolist(), "input_ids": input_list.tolist()}
            
            all_test_data.append(data_dict)
        
        return all_test_data
    
    @classmethod
    def build_gap_dataset(cls, vocab_size, num_examples, input_seq_len, num_kv_pairs, fixed_gap, insert_out_word=True, random_non_queries=True , random_seed=42):
        
        assert input_seq_len % 2 == 0, "input_seq_len must be even"
        # assert vocab_size > input_seq_len
        assert num_kv_pairs * 4 <= input_seq_len
        seed = random_seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        context_size = num_kv_pairs * 2
        # import pdb;pdb.set_trace()
        # create keys so that each key is present exactly once in each example
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)

        keys_unshuffled = np.tile(key_choices, (num_examples, 1))
        keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

        values_unshuffled = np.tile(value_choices, (num_examples, 1))
        values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

        # create sequences
        kvs = np.zeros((num_examples, input_seq_len), dtype=np.int64)
        kvs[:, 0:num_kv_pairs*2:2] = keys
        kvs[:, 1:num_kv_pairs*2:2] = values
        kvs[:,0+ fixed_gap::2] = keys

        labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
        labels[:,1+ fixed_gap::2] = values

        inputs, labels = torch.tensor(kvs[:, :]), torch.tensor(labels[:, 1:])
        # replace all the 0 with random values
        if random_non_queries:                          # 随机插入非 key-value值
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size+1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

        all_test_data = []

        for i in range(inputs.size(0)):  
            input_list = inputs[i].to(torch.int32)
            # label_idx = torch.nonzero(labels[i] != -100).flatten().to(torch.int32)
            # label_value = torch.index_select(labels[i], 0, label_idx).to(torch.int32)
            
            # data_dict = {'input': input_list, 'label_idx': label_idx, 'label_value': label_value}
            label_list = labels[i].to(torch.int32)

            data_dict = {'input': input_list, 'label': label_list}

            all_test_data.append(data_dict)
        
        return all_test_data
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        item = self.content[index]
        input_ids = item['input']
        label = item['label']
        attention_mask = torch.ones(input_ids.shape,dtype=input_ids.dtype)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label" : label
        }

        res.update(item)

        return res

if __name__ == '__main__':
   
    VOCAB_SIZE = 8192
    # def allocate_num_examples(num_key_pairs):
    #     """
    #     分配num_examples给每个key_pair，使得总和为800，并且每个分配是100的整数倍。
    #     """
    #     # import pdb;pdb.set_trace()
    #     allocations = [8//num_key_pairs *100] * num_key_pairs
    #     total_allocated = sum(allocations)
    #     i = num_key_pairs - 1  # 从最后一个配置开始分配

    #     # 从后往前分配，直到总分配量达到800
    #     while total_allocated < 800:
    #         allocations[i] += 100  # 每次增加100
    #         total_allocated += 100
    #         i -= 1  # 向前移动
            
    #     return allocations

    # # 存储每个配置的num_examples
    # num_examples_list = []

    # for gap in [2**i for i in range(2, 15)]:
    #     valid_key_pairs = [key_pair for key_pair in [2, 4, 8, 16, 32, 64, 128, 256] if gap >= key_pair * 2]
    
    #     # 根据有效的key_pair数量分配num_examples
    #     allocations = allocate_num_examples(len(valid_key_pairs))
        
    #     for key_pair, num_examples in zip(valid_key_pairs, allocations):
    #         input_seq_len = gap + 2 * key_pair
    #         test_data = MQARDataset.build_gap_dataset(
    #                     vocab_size=8192,    
    #                     input_seq_len=input_seq_len,
    #                     num_kv_pairs=key_pair,
    #                     num_examples=num_examples,
    #                     fixed_gap=gap, 
    #                     insert_out_word=False
    #                 )
    #         test_data_path = f"/public/home/ljt/tzc/data/MQAR/fixed_gaps/test_C8192_D{key_pair}_gap{gap}_in.pkl"
    #         auto_save_data(test_data,test_data_path)
  
  
  
  
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
    all_train_data = []
    all_test_data = []
    VOCAB_SIZE = 8192
    # train_configs = [
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 15000, "num_kv_pairs": 4},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 128, "num_examples": 15000, "num_kv_pairs": 8},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 5000, "num_kv_pairs": 16},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 5000, "num_kv_pairs": 32},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 5000, "num_kv_pairs": 64},
    #      {"vocab_size": VOCAB_SIZE, "input_seq_len": 512, "num_examples": 5000, "num_kv_pairs": 16},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 512, "num_examples": 5000, "num_kv_pairs": 32},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 512, "num_examples": 5000, "num_kv_pairs": 64},
    
    # ]
    test_configs = [
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 100, "num_kv_pairs": 4},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 100, "num_kv_pairs": 8},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 100, "num_kv_pairs": 16},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 128, "num_examples": 100, "num_kv_pairs": 8},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 128, "num_examples": 100, "num_kv_pairs": 16},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 128, "num_examples": 100, "num_kv_pairs": 32},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 100, "num_kv_pairs": 16},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 100, "num_kv_pairs": 32},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 100, "num_kv_pairs": 64},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 512, "num_examples": 100, "num_kv_pairs": 32},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 512, "num_examples": 100, "num_kv_pairs": 64},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 512, "num_examples": 100, "num_kv_pairs": 128},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 1024, "num_examples": 100, "num_kv_pairs": 64},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 1024, "num_examples": 100, "num_kv_pairs": 128},
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 1024, "num_examples": 100, "num_kv_pairs": 256},
    ]

    # for train_config in train_configs:
    #     train_data = MQARDataset.build_dataset(
    #                 vocab_size=8192, 
    #                 input_seq_len=train_config["input_seq_len"],
    #                 num_kv_pairs=train_config["num_kv_pairs"],
    #                 num_examples=train_config["num_examples"],
    #                 power_a=0.01,
    #                 tokenizer=None,
    #                 insert_out_word=False
    #             )
    #     all_train_data.extend(train_data)
    # # import pdb;pdb.set_trace()
    # train_data_path = "/public/home/ljt/tzc/data/MQAR/train_based_tzc.jsonl"
    # auto_save_data(all_train_data,train_data_path)
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
                    insert_out_word=False,
                    random_seed=5678
                )
        all_test_data.extend(test_data)
    test_data_path = f"/public/home/ljt/tzc/data/MQAR/test_based_in_metadata_1k_100.pkl"
    auto_save_data(all_test_data,test_data_path)