from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import random
import numpy as np
import torch
import glob

def build_dataset(configs, split='train', dataset_version='v5', mask_value=True, version_name=None, data_name=None, comment= None):
    all_data = []
    root_data_path = f"/nvme1/zecheng/data/MQAR/mqar-{dataset_version}-{version_name}-{data_name}" if not comment else \
                    f"/nvme1/zecheng/data/MQAR/mqar-{dataset_version}-{version_name}-{data_name}-{comment}"

    os.makedirs(root_data_path, exist_ok=True)
    data_path = f"{root_data_path}/{split}.jsonl"
    log_c(data_path)

    # if os.path.exists(data_path):
    #     log_c("data file exist, please check")
    #     exit()

    if dataset_version == 'v0':
        data_module = MQARDataset.build_standard_mqar
    if dataset_version == 'v0_shuffle':
        data_module = MQARDataset.build_shuffle_mqar
    elif dataset_version == 'v0_last':
        data_module = MQARDataset.build_last_mqar
    elif dataset_version == 'v6':
        data_module = MQARDataset.build_v6_dataset
    elif dataset_version == 'v6_standard':
        data_module = MQARDataset.build_v6_standard_dataset

    log_c(f"Using {data_module}")
    random_seed = {'train': 42, 'valid': 1234, 'test': 4567}
    
    with open(f"{root_data_path}/info", "a+") as f:
        f.write(str(split)+"\n")
        with tqdm(total=len(configs)) as pbar:
            for idx, config in enumerate(configs):

                key_len = config["key_len"]
                value_len = config["value_len"]
                context_len = config["context_len"]
                input_seq_len = config["input_seq_len"]
                num_kv_pairs = config["num_kv_pairs"]
                num_examples = config["num_examples"]
                mask_value = False if "v0" in dataset_version else True    
                add_key_loss = False  #True if split=="train" and dataset_version!="v0" else False
                add_spe_loss = True if split=="train" and "v0" not in dataset_version else False
                # add_key_loss = False
                data = data_module(
                    input_seq_len=input_seq_len,
                    num_kv_pairs=num_kv_pairs,
                    num_examples=num_examples,
                    key_length=key_len,
                    value_length=value_len,
                    context_len = context_len,
                    mask_value = mask_value ,
                    random_seed = random_seed[split] + idx,
                    add_key_loss=add_key_loss,
                    add_spe_loss=add_spe_loss,
                    random_non_queries= False         #False if 'nonoise' in dataset_version else True ,
                )
                log_f = f"vocab_size: {VOCAB_SIZE} input_seq_len: {input_seq_len} num_kv_pairs: {num_kv_pairs}\
                    key_length: {key_len} value_length: {value_len} context_len: {context_len} mask_value:{mask_value}\
                    add_key_loss: {add_key_loss} random_seed: {random_seed[split] + idx} num_examples: {num_examples}\
                    add_spe_loss: {add_spe_loss}"
                log_c(log_f)
                f.write(log_f + "\n")
                pbar.update(1)
                all_data.extend(data)
        f.write(str(all_data[0]) + "\n")
    print("example_data: \n", all_data[0])
    print("example_data: \n", all_data[len(all_data)//2])
    print("example_data: \n", all_data[-1])
    auto_save_data(all_data, data_path)


def generate_configs(max_seq_len, num_examples=None, split='train', dataset_version='v0', key_len=1, value_len=1, input_kv = None, VOCAB_SIZE=20000):
    configs = []
    if not num_examples:
        num_examples = 10000 if split=='train' else 100
    
    if max_seq_len is None:
        if split == "train":
            max_seq_len = 512
        if split == "valid":
            max_seq_len = 512
        if split == "test":
            max_seq_len = 512

    if input_kv is None:
        # for input_i in range(6, 20):
        #     for kvpair_x in range(0, input_i):
                # input_seq_len = 2 ** input_i
                # num_kv_pairs = 2 ** kvpair_x
        for input_i in range(1, 65, 2):
            
            input_seq_len =  input_i * 1000 
            num_kv_pairs = 1

            if input_seq_len > max_seq_len:
                break 

            if 'v0' in dataset_version:
                sp_token = 0
            else: 
                sp_token = 3

            single_kv_pair = key_len + value_len + sp_token

            if single_kv_pair * 2 * num_kv_pairs > input_seq_len:
                break

            configs.append({  
                "vocab_size": VOCAB_SIZE, 
                "input_seq_len": input_seq_len, 
                "num_examples": num_examples, 
                "num_kv_pairs": num_kv_pairs,
                "key_len": key_len, 
                "value_len":value_len,    
                "context_len": None})
            
            log_c(f"add_config: vocab_size_{VOCAB_SIZE} input_seq_len_{input_seq_len} num_kv_pairs_{num_kv_pairs}")

    else:
        for inlen, kv_pairs in input_kv:
            configs.append({  
                "vocab_size": VOCAB_SIZE, 
                "input_seq_len": inlen, 
                "num_examples": num_examples, 
                "num_kv_pairs": kv_pairs,
                "key_len": key_len, 
                "value_len":value_len,    
                "context_len": None})
            
            log_c(f"add_config: vocab_size_{VOCAB_SIZE} input_seq_len_{input_seq_len} num_kv_pairs_{num_kv_pairs}")


    for config in configs:
        print(config)

    log_c(f"single_kv_pair = {single_kv_pair}")
    log_c(f"total_configs_number = {len(configs)}")
    log_c(f"number_examples for per config = {num_examples}")
    log_c(f"total_data_number = {len(configs) * num_examples}")
    return configs 

def generate_test_configs(max_seq_len, num_examples=None, split='train', dataset_version='v5', key_len=1, value_len=1, context_len=None, max_kv_pairs=None, VOCAB_SIZE=20000):
    configs = []
    if not num_examples:
        num_examples = 100
    
    if not max_seq_len:
        max_seq_len = 512

    for kvpair_x in range(0, max_seq_len//2):
        input_seq_len = max_seq_len
        num_kv_pairs = 2 ** kvpair_x

        if input_seq_len > max_seq_len:
            break 
        
        if max_kv_pairs and num_kv_pairs > max_kv_pairs:
            break

        if 'v0' in dataset_version:
            sp_token = 0
        else: 
            sp_token = 3

        single_kv_pair = key_len + value_len + sp_token

        if single_kv_pair * 2 * num_kv_pairs > input_seq_len:
            break

        configs.append({  
            "vocab_size": VOCAB_SIZE, 
            "input_seq_len": input_seq_len, 
            "num_examples": num_examples, 
            "num_kv_pairs": num_kv_pairs,
            "key_len": key_len, 
            "value_len":value_len,    
            "context_len": context_len})
        
        log_c(f"add_config: vocab_size_{VOCAB_SIZE} input_seq_len_{input_seq_len} num_kv_pairs_{num_kv_pairs}")

        # if split == 'test':
        #     num_kv_pairs = (2 ** kvpair_x + 2 ** (kvpair_x+1)) //2
        #     configs.append({  
        #         "vocab_size": VOCAB_SIZE, 
        #         "input_seq_len": input_seq_len, 
        #         "num_examples": num_examples, 
        #         "num_kv_pairs": num_kv_pairs,
        #         "key_len": key_len, 
        #         "value_len":value_len,
        #         "context_len": context_len     })
        #     log_c(f"add_config: vocab_size_{VOCAB_SIZE} input_seq_len_{input_seq_len} num_kv_pairs_{num_kv_pairs}")

    for config in configs:
        print(config)

    log_c(f"single_kv_pair = {single_kv_pair}")
    log_c(f"total_configs_number = {len(configs)}")
    log_c(f"number_examples for per config = {num_examples}")
    log_c(f"total_data_number = {len(configs) * num_examples}")
    return configs 


class MQARDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs["max_seq_length"]
        self.cluster_batch = kwargs["cluster_batch"]

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

#################################################################
######     standard mqar                                 ########
######     kvkvkvkvkv ... k                              ########
#################################################################
    def build_standard_mqar(
            num_examples, input_seq_len, num_kv_pairs,  \
            key_length=1, value_length=1,               \
            context_len=1024, vocab_size=20000,         \
            power_a=0.01, random_seed=42,               \
            mask_value=True, random_non_queries=True, \
            add_key_loss=False, insert_out_word=False, \
            add_spe_loss=True):
        
        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

 
        context_size = num_kv_pairs * 2

        key_vocab_size = vocab_size // 2
        key_choices = np.arange(20, key_vocab_size)
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
        p = power_a * np.arange(1, space + 1) ** (power_a-1)
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

        labels = np.full((num_examples, input_seq_len ), -100, dtype=np.int64)
        np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

        inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, :])
        # import pdb;pdb.set_trace()
        # replace all the 0 with random values
        # mask = (inputs == 0)
        # if random_non_queries:
        #     for i in range(num_examples):
        #         row_mask = mask[i]  # Mask for the current row
        #         if row_mask.any():
        #             existing_values = set(inputs[i][inputs[i] != 0].tolist())
        #             num_zeros = row_mask.sum().item()
                    
        #             # Generate all possible values
        #             possible_values = torch.arange(20, vocab_size)
                    
        #             # Filter out existing values
        #             valid_values = possible_values[~torch.isin(possible_values, torch.tensor(list(existing_values)))]
                    
        #             # Check if there are enough valid values
        #             if valid_values.size(0) < num_zeros:
        #                 raise ValueError("Not enough valid values to replace zeros.")

        #             # Randomly sample values for replacement
        #             selected_values = valid_values[torch.randperm(valid_values.size(0))[:num_zeros]]
                    
        #             # Insert unique values into the output tensor
        #             inputs[i, row_mask] = selected_values
        if random_non_queries:
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size + 1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(20, vocab_size + 20, size=inputs.shape)[inputs == 0]


        all_test_data = []
        for i in range(inputs.size(0)):
            input_list = inputs[i].to(torch.int32)
            label_list = labels[i].to(torch.int32)
            non_neg100_indices = torch.nonzero(label_list != -100).squeeze()

            if not non_neg100_indices.shape:
                index_value_dict = {int(non_neg100_indices) : int(label_list[int(non_neg100_indices)])}
            else:
                index_value_dict = {int(index.item()) : label_list[index].item() for index in non_neg100_indices}
            data_dict = {'ctx_length': int(input_seq_len),'num_kv_pair':int(num_kv_pairs), 'key_len': int(key_length), 'value_len': int(value_length), 'V_id': index_value_dict, "input_ids": input_list.tolist()}
            # import pdb;pdb.set_trace()
            all_test_data.append(data_dict) 

        return all_test_data

    def build_last_mqar(
            num_examples, input_seq_len, num_kv_pairs,  \
            key_length=1, value_length=1,               \
            context_len=1024, vocab_size=20000,         \
            power_a=0.01, random_seed=42,               \
            mask_value=True, random_non_queries=True, \
            add_key_loss=False, insert_out_word=False, \
            add_spe_loss=True):
        
        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # import pdb;pdb.set_trace()
        # context_size = num_kv_pairs * 2
        context_size = input_seq_len // 2

        key_vocab_size = vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)

        keys_unshuffled = np.tile(key_choices, (num_examples, 1))
        keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

        values_unshuffled = np.tile(value_choices, (num_examples, 1))
        values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

        # create sequences
        kvs = np.zeros((num_examples, context_size), dtype=np.int64)
        kvs[:, (context_size-num_kv_pairs*2)::2] = keys
        kvs[:, (context_size-num_kv_pairs*2+1)::2] = values

        # compute power law
        space = (input_seq_len - context_size) // 2
        p = power_a * np.arange(1, space + 1) ** (power_a-1)
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

        inputs, labels = torch.tensor(examples[:, :]), torch.tensor(labels[:, :])
        
        # replace all the 0 with random values
        if random_non_queries:
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size + 1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(20, vocab_size + 20, size=inputs.shape)[inputs == 0]


        all_test_data = []
        for i in range(inputs.size(0)):
            input_list = inputs[i].to(torch.int32)
            label_list = labels[i].to(torch.int32)
            non_neg100_indices = torch.nonzero(label_list != -100).squeeze()

            if not non_neg100_indices.shape:
                index_value_dict = {int(non_neg100_indices) : int(label_list[int(non_neg100_indices)])}
            else:
                index_value_dict = {int(index.item()) : label_list[index].item() for index in non_neg100_indices}
            data_dict = {'ctx_length': int(input_seq_len),'num_kv_pair':int(num_kv_pairs), 'key_len': int(key_length), 'value_len': int(value_length), 'V_id': index_value_dict, "input_ids": input_list.tolist()}
            # import pdb;pdb.set_trace()
            all_test_data.append(data_dict) 

        return all_test_data
    
    def build_shuffle_mqar(
            num_examples, input_seq_len, num_kv_pairs,  \
            key_length=1, value_length=1,               \
            context_len=1024, vocab_size=20000,         \
            power_a=0.01, random_seed=42,               \
            mask_value=True, random_non_queries=True, \
            add_key_loss=False, insert_out_word=False, \
            add_spe_loss=True):
        
        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 

        # import pdb;pdb.set_trace()

        single_kv_pair_lens =  key_length + value_length 
        context_size = input_seq_len // 2
 
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(20, key_vocab_size + 20)
        value_choices = np.arange(key_vocab_size + 20, vocab_size + 20)

        keys = np.random.choice(key_choices, size=(num_examples, num_kv_pairs, key_length), replace=True)
        values = np.random.choice(value_choices, size=(num_examples, num_kv_pairs, value_length), replace=True)

        kvs = np.zeros((num_examples, input_seq_len // 2 ), dtype=np.int64)

        kv_space = input_seq_len // 2  // single_kv_pair_lens 
        kv_X = np.stack([np.arange(kv_space, dtype=int)] * num_examples)
        kv_gaps = np.apply_along_axis(np.random.choice, axis=1, arr=kv_X, replace=False, size=num_kv_pairs)
        kv_start_indices = kv_gaps * (single_kv_pair_lens)

        expanded_start_indices = kv_start_indices[:,:,None]  + np.arange(key_length) 
        expanded_start_indices_1 = kv_start_indices[:,:,None] + np.arange(key_length) + 1
        kvs[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys
        kvs[np.arange(num_examples)[:, None, None], expanded_start_indices_1] = values


        space = input_seq_len // 2 // single_kv_pair_lens  # Adjust for longer keys in queries
        p = power_a * np.arange(1, space + 1) ** (power_a - 1)
        p = p / p.sum()

        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
        queries = np.zeros((num_examples, input_seq_len - context_size ), dtype=np.int64)

        start_indices = gaps * (single_kv_pair_lens)
        expanded_start_indices = start_indices[:,:,None] + np.arange(key_length)  # 添加偏移量
        queries[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys

        examples = np.concatenate([kvs, queries], axis=1)

        labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)

        value_start_index = start_indices + context_size + key_length
        value_expanded_start_indices = value_start_index[:,:,None] + np.arange(value_length)  # 添加偏移量
        labels[np.arange(num_examples)[:, None, None], value_expanded_start_indices] = values

        inputs = torch.tensor(examples[:, :])

        if mask_value:
            mask = torch.Tensor(labels[:,:-1] != -100).bool()
            inputs[mask] = 12

        if add_key_loss and key_length>1:
            key_start_index = start_indices + context_size + 1 + 1
            key_expanded_start_indices = key_start_index[:,:,None] + np.arange(key_length-1)  # 添加偏移量
            # import pdb;pdb.set_trace()
            labels[np.arange(num_examples)[:, None, None], key_expanded_start_indices] = keys[:,:,1:]
        
        if add_spe_loss:
            # import pdb;pdb.set_trace()
            key_start_index = start_indices + context_size
            # labels[np.arange(num_examples)[:, None], key_start_index] = 10
            labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length + value_length + 1] = 10
            # labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length] = 11
            

        labels = torch.tensor(labels[:, :])
        if random_non_queries:
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size + 1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(20, vocab_size + 20, size=inputs.shape)[inputs == 0]


        all_test_data = []
        for i in range(inputs.size(0)):
            input_list = inputs[i].to(torch.int32)
            label_list = labels[i].to(torch.int32)
            non_neg100_indices = torch.nonzero(label_list != -100).squeeze()

            if not non_neg100_indices.shape:
                index_value_dict = {int(non_neg100_indices) : int(label_list[int(non_neg100_indices)])}
            else:

                index_value_dict = {int(index.item()) : label_list[index].item() for index in non_neg100_indices}
            data_dict = {'ctx_length': int(input_seq_len),'num_kv_pair':int(num_kv_pairs), 'key_len': int(key_length), 'value_len': int(value_length), 'V_id': index_value_dict, "input_ids": input_list.tolist()}
            # import pdb;pdb.set_trace()
            all_test_data.append(data_dict) 

        return all_test_data

    def build_robustness_standard_mqar(
            num_examples, input_seq_len, num_kv_pairs,  \
            key_length=1, value_length=1,               \
            context_len=1024, vocab_size=20000,         \
            power_a=0.01, random_seed=42,               \
            mask_value=True, random_non_queries=True, \
            add_key_loss=False, insert_out_word=False, \
            add_spe_loss=True):
        
        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # import pdb;pdb.set_trace()
        context_size = num_kv_pairs * 4

        key_vocab_size = vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)

        keys_unshuffled = np.tile(key_choices, (num_examples, 1))
        keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

        values_unshuffled = np.tile(value_choices, (num_examples, 1))
        values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

        # create sequences
        kvs = np.zeros((num_examples, context_size), dtype=np.int64)
        kvs[:, 0:num_kv_pairs * 2:2] = keys
        kvs[:, 1:num_kv_pairs * 2 + 1 :2] = values


        #####  insert noise  key-pairs  ######
        kvs[:, (num_kv_pairs * 2)::2] = keys
        kvs[:, (num_kv_pairs * 2+1)::2] = values + 1



        # compute power law
        space = (input_seq_len - context_size ) // 2
        p = power_a * np.arange(1, space + 1) ** (power_a-1)
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

        inputs, labels = torch.tensor(examples[:, :]), torch.tensor(labels[:, :])
        
        # replace all the 0 with random values
        if random_non_queries:
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size + 1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(20, vocab_size + 20, size=inputs.shape)[inputs == 0]


        all_test_data = []
        for i in range(inputs.size(0)):
            input_list = inputs[i].to(torch.int32)
            label_list = labels[i].to(torch.int32)
            non_neg100_indices = torch.nonzero(label_list != -100).squeeze()

            if not non_neg100_indices.shape:
                index_value_dict = {int(non_neg100_indices) : int(label_list[int(non_neg100_indices)])}
            else:
                index_value_dict = {int(index.item()) : label_list[index].item() for index in non_neg100_indices}
            data_dict = {'ctx_length': int(input_seq_len),'num_kv_pair':int(num_kv_pairs), 'key_len': int(key_length), 'value_len': int(value_length), 'V_id': index_value_dict, "input_ids": input_list.tolist()}
            # import pdb;pdb.set_trace()
            all_test_data.append(data_dict) 

        return all_test_data

    def build_robustness_last_mqar(
            num_examples, input_seq_len, num_kv_pairs,  \
            key_length=1, value_length=1,               \
            context_len=1024, vocab_size=20000,         \
            power_a=0.01, random_seed=42,               \
            mask_value=True, random_non_queries=True, \
            add_key_loss=False, insert_out_word=False, \
            add_spe_loss=True):
        
        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # import pdb;pdb.set_trace()
        context_size = input_seq_len//2

        key_vocab_size = vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)

        keys_unshuffled = np.tile(key_choices, (num_examples, 1))
        keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

        values_unshuffled = np.tile(value_choices, (num_examples, 1))
        values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

        # create sequences
        kvs = np.zeros((num_examples, context_size), dtype=np.int64)
        kvs[:, 0:num_kv_pairs * 2:2] = keys
        kvs[:, 1:num_kv_pairs * 2 + 1 :2] = values

        # import pdb;pdb.set_trace()
        #####  insert noise  key-pairs  ######
        kvs[:, (context_size - 2 * num_kv_pairs)::2] = keys
        kvs[:, (context_size - 2 * num_kv_pairs + 1)::2] = values + 1



        # compute power law
        space = (input_seq_len - context_size ) // 2
        p = power_a * np.arange(1, space + 1) ** (power_a-1)
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

        inputs, labels = torch.tensor(examples[:, :]), torch.tensor(labels[:, :])
        
        # replace all the 0 with random values
        if random_non_queries:
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size + 1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(20, vocab_size + 20, size=inputs.shape)[inputs == 0]


        all_test_data = []
        for i in range(inputs.size(0)):
            input_list = inputs[i].to(torch.int32)
            label_list = labels[i].to(torch.int32)
            non_neg100_indices = torch.nonzero(label_list != -100).squeeze()

            if not non_neg100_indices.shape:
                index_value_dict = {int(non_neg100_indices) : int(label_list[int(non_neg100_indices)])}
            else:
                index_value_dict = {int(index.item()) : label_list[index].item() for index in non_neg100_indices}
            data_dict = {'ctx_length': int(input_seq_len),'num_kv_pair':int(num_kv_pairs), 'key_len': int(key_length), 'value_len': int(value_length), 'V_id': index_value_dict, "input_ids": input_list.tolist()}
            # import pdb;pdb.set_trace()
            all_test_data.append(data_dict) 

        return all_test_data

    def build_robustness_split_mqar(
            num_examples, input_seq_len, num_kv_pairs,  \
            key_length=1, value_length=1,               \
            context_len=1024, vocab_size=20000,         \
            power_a=0.01, random_seed=42,               \
            mask_value=True, random_non_queries=True, \
            add_key_loss=False, insert_out_word=False, \
            add_spe_loss=True):
        
        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        block_number = 4 
        context_size = input_seq_len // 2
        kv_pair_context = num_kv_pairs * 2
        block_size = context_size // block_number

        key_vocab_size = vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)

        keys_unshuffled = np.tile(key_choices, (num_examples, 1))
        keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

        values_unshuffled = np.tile(value_choices, (num_examples, 1))
        values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

        # create sequences
        kvs = np.zeros((num_examples, context_size), dtype=np.int64)

        # import pdb;pdb.set_trace()
        for i in range (block_number):
            kvs[:, i * block_size:  i * block_size+ kv_pair_context       :2 ] = keys
            kvs[:, i * block_size + 1: i * block_size + 1 + kv_pair_context   :2] = values + i

        # compute power law
        space = (input_seq_len - context_size ) // 2
        p = power_a * np.arange(1, space + 1) ** (power_a-1)
        p = p / p.sum()

        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

        # queries and answers
        queries = np.zeros((num_examples, input_seq_len - context_size ), dtype=np.int64)
        np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
        examples = np.concatenate([
            kvs, 
            queries
        ], axis=1)

        labels = np.full((num_examples, input_seq_len ), -100, dtype=np.int64)
        np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

        inputs, labels = torch.tensor(examples[:, :]), torch.tensor(labels[:, :])
        
        # replace all the 0 with random values
        if random_non_queries:
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size + 1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(20, vocab_size + 20, size=inputs.shape)[inputs == 0]


        all_test_data = []
        for i in range(inputs.size(0)):
            input_list = inputs[i].to(torch.int32)
            label_list = labels[i].to(torch.int32)
            
            # import pdb;pdb.set_trace()
            non_neg100_indices = torch.nonzero(label_list != -100).squeeze()

            if not non_neg100_indices.shape:
                index_value_dict = {int(non_neg100_indices) : int(label_list[int(non_neg100_indices)])}
            else:
                index_value_dict = {int(index.item()) : label_list[index].item() for index in non_neg100_indices}
            data_dict = {'ctx_length': int(input_seq_len),'num_kv_pair':int(num_kv_pairs), 'key_len': int(key_length), 'value_len': int(value_length), 'V_id': index_value_dict, "input_ids": input_list.tolist()}
            # import pdb;pdb.set_trace()
            all_test_data.append(data_dict) 

        return all_test_data
    @classmethod
    def build_gap_dataset(  num_examples, input_seq_len, num_kv_pairs,  \
                            key_length=1, value_length=1,               \
                            context_len=1024, vocab_size=20000,         \
                            power_a=0.01, random_seed=42,               \
                            mask_value=True, random_non_queries=True, \
                            insert_out_word=False, add_spe_token=True):
        
        assert input_seq_len % 2 == 0, "input_seq_len must be even"
        # assert vocab_size > input_seq_len
        assert num_kv_pairs * 4 <= input_seq_len
        seed = random_seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        context_size = num_kv_pairs * 2
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
    

    def build_v6_dataset(   num_examples, input_seq_len, num_kv_pairs,  \
                            key_length=1, value_length=1,               \
                            context_len=1024, vocab_size=20000,         \
                            power_a=0.01, random_seed=42,               \
                            mask_value=True, random_non_queries=True,   \
                            insert_out_word=False, add_key_loss=False,   \
                            add_spe_loss=False):
        assert input_seq_len % 2 == 0, "input_seq_len must be even"
        assert num_kv_pairs * ( key_length + value_length + 3 )  <= input_seq_len // 2  # Adjust for longer keys and values

        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        single_kv_pair_lens = (3 + key_length + value_length)   
        context_size = input_seq_len // 2
 
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(20, key_vocab_size + 20)
        value_choices = np.arange(key_vocab_size + 20, vocab_size + 20)

        keys = np.random.choice(key_choices, size=(num_examples, num_kv_pairs, key_length), replace=True)
        values = np.random.choice(value_choices, size=(num_examples, num_kv_pairs, value_length), replace=True)

        kvs = np.zeros((num_examples, input_seq_len // 2 ), dtype=np.int64)

        kv_space = input_seq_len // 2  // single_kv_pair_lens 
        kv_X = np.stack([np.arange(kv_space, dtype=int)] * num_examples)
        kv_gaps = np.apply_along_axis(np.random.choice, axis=1, arr=kv_X, replace=False, size=num_kv_pairs)
        kv_start_indices = kv_gaps * (single_kv_pair_lens)

        kvs[np.arange(num_examples)[:, None], kv_start_indices] = 10
        kvs[np.arange(num_examples)[:, None], kv_start_indices + 1 + key_length + value_length + 1] = 10
        kvs[np.arange(num_examples)[:, None], kv_start_indices + 1 + key_length] = 11
        expanded_start_indices = kv_start_indices[:,:,None] + 1 + np.arange(key_length)  # 添加偏移量
        expanded_start_indices_1 = kv_start_indices[:,:,None] + 2 + key_length +  np.arange(value_length) # 添加偏移量
        kvs[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys
        kvs[np.arange(num_examples)[:, None, None], expanded_start_indices_1] = values

        space = input_seq_len // 2 // single_kv_pair_lens  # Adjust for longer keys in queries
        p = power_a * np.arange(1, space + 1) ** (power_a - 1)
        p = p / p.sum()

        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
        queries = np.zeros((num_examples, input_seq_len - context_size ), dtype=np.int64)

        start_indices = gaps * (single_kv_pair_lens)
        queries[np.arange(num_examples)[:, None], start_indices] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length + value_length + 1] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length] = 11
        expanded_start_indices = start_indices[:,:,None] + 1 + np.arange(key_length)  # 添加偏移量
        queries[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys

        examples = np.concatenate([kvs, queries], axis=1)

        labels = np.full((num_examples, input_seq_len ), -100, dtype=np.int64)

        value_start_index = start_indices + context_size + 2 + key_length
        value_expanded_start_indices = value_start_index[:,:,None] + np.arange(value_length)  # 添加偏移量
        labels[np.arange(num_examples)[:, None, None], value_expanded_start_indices] = values

        inputs = torch.tensor(examples[:, :])

        if mask_value:
            mask = torch.Tensor(labels[:,:] != -100).bool()
            inputs[mask] = 12

        if add_key_loss and key_length>1:
            key_start_index = start_indices + context_size + 1 + 1
            key_expanded_start_indices = key_start_index[:,:,None] + np.arange(key_length-1)  # 添加偏移量
            # import pdb;pdb.set_trace()
            labels[np.arange(num_examples)[:, None, None], key_expanded_start_indices] = keys[:,:,1:]
        
        if add_spe_loss:
            # import pdb;pdb.set_trace()
            key_start_index = start_indices + context_size
            # labels[np.arange(num_examples)[:, None], key_start_index] = 10
            labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length + value_length + 1] = 10
            # labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length] = 11
            

        labels = torch.tensor(labels[:, :])
        if random_non_queries:
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size + 1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(20, vocab_size + 20, size=inputs.shape)[inputs == 0]


        all_test_data = []
        for i in range(inputs.size(0)):
            input_list = inputs[i].to(torch.int32)
            label_list = labels[i].to(torch.int32)
            non_neg100_indices = torch.nonzero(label_list != -100).squeeze()

            if not non_neg100_indices.shape:
                index_value_dict = {int(non_neg100_indices) : int(label_list[int(non_neg100_indices)])}
            else:
                index_value_dict = {int(index.item()) : label_list[index].item() for index in non_neg100_indices}
            data_dict = {'ctx_length': int(input_seq_len),'num_kv_pair':int(num_kv_pairs), 'key_len': int(key_length), 'value_len': int(value_length), 'V_id': index_value_dict, "input_ids": input_list.tolist()}
            # import pdb;pdb.set_trace()
            all_test_data.append(data_dict) 

        return all_test_data


#################################################################
######     v6  格式上与v2相同<s1> key <s2> value <s1>     ########
######     query中位置用特殊字符mask代替(使用12),          ########
######     其他context位置还是插入随机值                   ########
#################################################################
    def build_v6_standard_dataset(   num_examples, input_seq_len, num_kv_pairs,  \
                            key_length=1, value_length=1,               \
                            context_len=1024, vocab_size=20000,         \
                            power_a=0.01, random_seed=42,               \
                            mask_value=True, random_non_queries=True,   \
                            insert_out_word=False, add_key_loss=False,   \
                            add_spe_loss=False):
        assert input_seq_len % 2 == 0, "input_seq_len must be even"
        assert num_kv_pairs * ( key_length + value_length + 3 )  <= input_seq_len // 2  # Adjust for longer keys and values

        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        single_kv_pair_lens = (3 + key_length + value_length)
        kvpairs_size = num_kv_pairs * single_kv_pair_lens
        context_size = kvpairs_size
 
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(20, key_vocab_size + 20)
        value_choices = np.arange(key_vocab_size + 20, vocab_size + 20)

        keys = np.random.choice(key_choices, size=(num_examples, num_kv_pairs, key_length), replace=True)
        values = np.random.choice(value_choices, size=(num_examples, num_kv_pairs, value_length), replace=True)

        # import pdb;pdb.set_trace()
        kvs = np.zeros((num_examples, kvpairs_size), dtype=np.int64)
        for i in range(num_kv_pairs):
            kvs[:, single_kv_pair_lens * i] = 10  # <s1>
            kvs[:, single_kv_pair_lens * i + 1:single_kv_pair_lens * i + 1 + key_length] = keys[:, i, :]
            kvs[:, single_kv_pair_lens * i + 1 + key_length] = 11  # <s2>
            kvs[:, single_kv_pair_lens * i + 2 + key_length:single_kv_pair_lens * i + 2 + key_length + value_length] = values[:, i, :]
            kvs[:, single_kv_pair_lens * i + 2 + key_length + value_length] = 10  # <s1>

        # import pdb;pdb.set_trace()
        space = (input_seq_len - kvpairs_size) // single_kv_pair_lens  # Adjust for longer keys in queries
        p = power_a * np.arange(1, space + 1) ** (power_a - 1)
        p = p / p.sum()


        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
        queries = np.zeros((num_examples, input_seq_len - kvpairs_size ), dtype=np.int64)

        start_indices = gaps * (single_kv_pair_lens)
        queries[np.arange(num_examples)[:, None], start_indices] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length + value_length + 1] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length] = 11
        expanded_start_indices = start_indices[:,:,None] + 1 + np.arange(key_length)  # 添加偏移量
        queries[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys

        examples = np.concatenate([kvs, queries], axis=1)

        labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)

        value_start_index = start_indices + kvpairs_size + 2 + key_length
        value_expanded_start_indices = value_start_index[:,:,None] + np.arange(value_length)  # 添加偏移量
        labels[np.arange(num_examples)[:, None, None], value_expanded_start_indices] = values

        inputs = torch.tensor(examples[:, :])

        if mask_value:
            mask = torch.Tensor(labels[:,:-1] != -100).bool()
            inputs[mask] = 12

        if add_key_loss and key_length>1:
            key_start_index = start_indices + context_size + 1 + 1
            key_expanded_start_indices = key_start_index[:,:,None] + np.arange(key_length-1)  # 添加偏移量
            # import pdb;pdb.set_trace()
            labels[np.arange(num_examples)[:, None, None], key_expanded_start_indices] = keys[:,:,1:]
        
        if add_spe_loss:
            # import pdb;pdb.set_trace()
            key_start_index = start_indices + context_size
            # labels[np.arange(num_examples)[:, None], key_start_index] = 10
            labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length + value_length + 1] = 10
            # labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length] = 11
            

        labels = torch.tensor(labels[:, :])
        if random_non_queries:
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size + 1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(20, vocab_size + 20, size=inputs.shape)[inputs == 0]


        all_test_data = []
        for i in range(inputs.size(0)):
            input_list = inputs[i].to(torch.int32)
            label_list = labels[i].to(torch.int32)
            non_neg100_indices = torch.nonzero(label_list != -100).squeeze()

            if not non_neg100_indices.shape:
                index_value_dict = {int(non_neg100_indices) : int(label_list[int(non_neg100_indices)])}
            else:
                index_value_dict = {int(index.item()) : label_list[index].item() for index in non_neg100_indices}
            data_dict = {'ctx_length': int(input_seq_len),'num_kv_pair':int(num_kv_pairs), 'key_len': int(key_length), 'value_len': int(value_length), 'V_id': index_value_dict, "input_ids": input_list.tolist()}
            # import pdb;pdb.set_trace()
            all_test_data.append(data_dict) 

        return all_test_data
    

    def build_v6_standard_robustness_dataset(   num_examples, input_seq_len, num_kv_pairs,  \
                            key_length=1, value_length=1,               \
                            context_len=1024, vocab_size=20000,         \
                            power_a=0.01, random_seed=42,               \
                            mask_value=True, random_non_queries=True,   \
                            insert_out_word=False, add_key_loss=False,   \
                            add_spe_loss=False):
        assert input_seq_len % 2 == 0, "input_seq_len must be even"
        assert num_kv_pairs * ( key_length + value_length + 3 )  <= input_seq_len // 2  # Adjust for longer keys and values

        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        single_kv_pair_lens = (3 + key_length + value_length)
        kvpairs_size = num_kv_pairs * single_kv_pair_lens
        context_size = kvpairs_size
 
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(20, key_vocab_size + 20)
        value_choices = np.arange(key_vocab_size + 20, vocab_size + 20)

        keys1 = np.random.choice(key_choices, size=(num_examples, num_kv_pairs, key_length-1), replace=True)
        keys2 = np.random.choice(key_choices, size=(num_examples, 1), replace=True)
        keys2_expanded = np.expand_dims(keys2, axis=1)
        keys2_broadcasted = np.broadcast_to(keys2_expanded, (num_examples, num_kv_pairs, key_length-1))
        keys = np.concatenate((keys1, keys2_broadcasted), axis=2)
        values = np.random.choice(value_choices, size=(num_examples, num_kv_pairs, value_length), replace=True)

        # import pdb;pdb.set_trace()
        kvs = np.zeros((num_examples, kvpairs_size), dtype=np.int64)
        for i in range(num_kv_pairs):
            kvs[:, single_kv_pair_lens * i] = 10  # <s1>
            kvs[:, single_kv_pair_lens * i + 1:single_kv_pair_lens * i + 1 + key_length] = keys[:, i, :]
            kvs[:, single_kv_pair_lens * i + 1 + key_length] = 11  # <s2>
            kvs[:, single_kv_pair_lens * i + 2 + key_length:single_kv_pair_lens * i + 2 + key_length + value_length] = values[:, i, :]
            kvs[:, single_kv_pair_lens * i + 2 + key_length + value_length] = 10  # <s1>

        # import pdb;pdb.set_trace()
        space = (input_seq_len - kvpairs_size) // single_kv_pair_lens  # Adjust for longer keys in queries
        p = power_a * np.arange(1, space + 1) ** (power_a - 1)
        p = p / p.sum()


        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=None, size=num_kv_pairs)
        queries = np.zeros((num_examples, input_seq_len - kvpairs_size ), dtype=np.int64)

        start_indices = gaps * (single_kv_pair_lens)
        queries[np.arange(num_examples)[:, None], start_indices] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length + value_length + 1] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length] = 11
        expanded_start_indices = start_indices[:,:,None] + 1 + np.arange(key_length)  # 添加偏移量
        queries[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys

        examples = np.concatenate([kvs, queries], axis=1)

        labels = np.full((num_examples, input_seq_len ), -100, dtype=np.int64)

        value_start_index = start_indices + kvpairs_size + 2 + key_length
        value_expanded_start_indices = value_start_index[:,:,None] + np.arange(value_length)  # 添加偏移量
        labels[np.arange(num_examples)[:, None, None], value_expanded_start_indices] = values

        inputs = torch.tensor(examples[:, :])

        if mask_value:
            mask = torch.Tensor(labels[:,:] != -100).bool()
            inputs[mask] = 12

        if add_key_loss and key_length>1:
            key_start_index = start_indices + context_size + 1 + 1
            key_expanded_start_indices = key_start_index[:,:,None] + np.arange(key_length-1)  # 添加偏移量
            # import pdb;pdb.set_trace()
            labels[np.arange(num_examples)[:, None, None], key_expanded_start_indices] = keys[:,:,1:]
        
        if add_spe_loss:
            # import pdb;pdb.set_trace()
            key_start_index = start_indices + context_size
            # labels[np.arange(num_examples)[:, None], key_start_index] = 10
            labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length + value_length + 1] = 10
            # labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length] = 11
            

        labels = torch.tensor(labels[:, :])
        if random_non_queries:
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size + 1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(20, vocab_size + 20, size=inputs.shape)[inputs == 0]


        all_test_data = []
        for i in range(inputs.size(0)):
            input_list = inputs[i].to(torch.int32)
            label_list = labels[i].to(torch.int32)
            non_neg100_indices = torch.nonzero(label_list != -100).squeeze()
            # import pdb;pdb.set_trace()
            if not non_neg100_indices.shape:
                index_value_dict = {int(non_neg100_indices) : int(label_list[int(non_neg100_indices)])}
            else:
                index_value_dict = {int(index.item()) : label_list[index].item() for index in non_neg100_indices}
            data_dict = {'ctx_length': int(input_seq_len),'num_kv_pair':int(num_kv_pairs), 'key_len': int(key_length), 'value_len': int(value_length), 'V_id': index_value_dict, "input_ids": input_list.tolist()}
            # import pdb;pdb.set_trace()
            all_test_data.append(data_dict) 

        return all_test_data



    def build_v7_dataset(  num_examples, input_seq_len, num_kv_pairs,  \
                        key_length=1, value_length=1,               \
                        context_len=1024, vocab_size=20000,         \
                        power_a=0.01, random_seed=42,               \
                        mask_value=True, random_non_queries=True,   \
                        insert_out_word=False, add_key_loss=False,   \
                        add_spe_loss=False, split='Train'):
        assert input_seq_len % 2 == 0, "input_seq_len must be even"
        assert num_kv_pairs * ( key_length + value_length + 3 )  <= input_seq_len // 2  # Adjust for longer keys and values

        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        single_kv_pair_lens = (3 + key_length + value_length)   
        context_size = input_seq_len // 2

        key_vocab_size = vocab_size // 2
        key_choices = np.arange(20, key_vocab_size + 20)
        value_choices = np.arange(key_vocab_size + 20, vocab_size + 20)

        keys = np.random.choice(key_choices, size=(num_examples, num_kv_pairs, key_length), replace=True)
        values = np.random.choice(value_choices, size=(num_examples, num_kv_pairs, value_length), replace=True)

        kvs = np.zeros((num_examples, input_seq_len // 2 ), dtype=np.int64)

        kv_space = input_seq_len // 2  // single_kv_pair_lens 
        kv_X = np.stack([np.arange(kv_space, dtype=int)] * num_examples)
        kv_gaps = np.apply_along_axis(np.random.choice, axis=1, arr=kv_X, replace=False, size=num_kv_pairs)
        kv_start_indices = kv_gaps * (single_kv_pair_lens)

        kvs[np.arange(num_examples)[:, None], kv_start_indices] = 10
        kvs[np.arange(num_examples)[:, None], kv_start_indices + 1 + key_length + value_length + 1] = 10
        kvs[np.arange(num_examples)[:, None], kv_start_indices + 1 + key_length] = 11
        expanded_start_indices = kv_start_indices[:,:,None] + 1 + np.arange(key_length)  # 添加偏移量
        expanded_start_indices_1 = kv_start_indices[:,:,None] + 2 + key_length +  np.arange(value_length) # 添加偏移量
        kvs[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys
        kvs[np.arange(num_examples)[:, None, None], expanded_start_indices_1] = values

        space = input_seq_len // 2 // single_kv_pair_lens  # Adjust for longer keys in queries
        p = power_a * np.arange(1, space + 1) ** (power_a - 1)
        p = p / p.sum()

        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
        queries = np.zeros((num_examples, input_seq_len - context_size ), dtype=np.int64)

        start_indices = gaps * (single_kv_pair_lens)
        queries[np.arange(num_examples)[:, None], start_indices] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length + value_length + 1] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length] = 11
        expanded_start_indices = start_indices[:,:,None] + 1 + np.arange(key_length)  # 添加偏移量
        queries[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys

        examples = np.concatenate([kvs, queries], axis=1)

        labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)

        value_start_index = start_indices + context_size + 2 + key_length
        value_expanded_start_indices = value_start_index[:,:,None] + np.arange(value_length)  # 添加偏移量
        labels[np.arange(num_examples)[:, None, None], value_expanded_start_indices] = values

        inputs = torch.tensor(examples[:, :])

        if mask_value:
            mask = torch.Tensor(labels[:,:-1] != -100).bool()
            inputs[mask] = 12

        if add_key_loss and key_length>1:
            key_start_index = start_indices + context_size + 1 + 1
            key_expanded_start_indices = key_start_index[:,:,None] + np.arange(key_length-1)  # 添加偏移量
            # import pdb;pdb.set_trace()
            labels[np.arange(num_examples)[:, None, None], key_expanded_start_indices] = keys[:,:,1:]
        
        if add_spe_loss:
            # import pdb;pdb.set_trace()
            key_start_index = start_indices + context_size
            # labels[np.arange(num_examples)[:, None], key_start_index] = 10
            labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length + value_length + 1] = 10
            # labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length] = 11
            

        # import pdb;pdb.set_trace()
        labels = torch.tensor(labels[:, :])
        last_labels = labels[labels!=-100].view(labels.size(0), -1)
        # import pdb;pdb.set_trace()
        if split=='test':
            mask_last_label = last_labels.clone()
            mask_last_label[last_labels!=10] = 12

        Bos = np.full((num_examples, 1), 15, dtype=np.int64)
        Eos = np.full((num_examples, 1), 16, dtype=np.int64)
        if split=='test' :
            inputs = np.concatenate([inputs, Bos, mask_last_label, Eos], axis=1)
        else:
            inputs = np.concatenate([inputs, Bos, last_labels, Eos], axis=1)
        inputs = torch.tensor(inputs[:, :])
        fin_labels = np.full((num_examples, input_seq_len), -100, dtype=np.int64)
        fin_labels = np.concatenate([fin_labels, Bos, last_labels, Eos], axis=1)
        fin_labels = torch.tensor(fin_labels[:, :])


        if random_non_queries:
            if insert_out_word:
                inputs[inputs == 0] = torch.randint(vocab_size + 1, 20480, size=inputs.shape)[inputs == 0]
            else:
                inputs[inputs == 0] = torch.randint(20, vocab_size + 20, size=inputs.shape)[inputs == 0]

        inputs = inputs.to(torch.int32)
        all_test_data = []
        for i in range(inputs.size(0)):
            input_list = inputs[i].to(torch.int32)
            label_list = fin_labels[i].to(torch.int32)
            non_neg100_indices = torch.nonzero(label_list != -100).squeeze()

            if not non_neg100_indices.shape:
                index_value_dict = {int(non_neg100_indices) : int(label_list[int(non_neg100_indices)])}
            else:
                index_value_dict = {int(index.item()) : label_list[index].item() for index in non_neg100_indices}
            data_dict = {'ctx_length': int(input_seq_len),'num_kv_pair':int(num_kv_pairs), 'key_len': int(key_length), 'value_len': int(value_length), 'V_id': index_value_dict, "input_ids": input_list.tolist()}
            # import pdb;pdb.set_trace()
            all_test_data.append(data_dict) 

        return all_test_data


if __name__ == '__main__':
    ##  train_seed  42 + idx
    ##  valid_seed  1234 + idx
    ##  test_seed   4567 + idx


    # train_configs = [    
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64),
#     ]
#     test_configs = [
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=4),
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=8),
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=16),
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
#         MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
# ]
    

    # configs = generate_configs(num_examples=20000, split='train', dataset_version='v0', input_kv=[(64,4),(128,8),(256,16),(256,32),(256,64)], VOCAB_SIZE=8192)
    
    # VOCAB_SIZE = 8192
    # all_data = []
    # train_input_kv=[(64,4),(128,8),(256,16),(256,32),(256,64)]
    # valid_input_kv=[(64,4),(64,8),(128,4),(128,8),(128,16),(256,4),(256,8),(256,16),(256,32),(256,64)]
    # test_input_kv=[(64,4),(64,8),(128,4),(128,8),(128,16),(256,4),(256,8),(256,16),(256,32),(256,64)]
    # split = "train"
    # for input_l, kv in train_input_kv:
    #     data = MQARDataset.build_standard_mqar(
    #             input_seq_len=input_l,
    #             num_kv_pairs=kv,
    #             num_examples=20000 if input_l!=64 else 100000,
    #             key_length=1,
    #             value_length=1,
    #             context_len = None,
    #             mask_value = False ,
    #             random_seed = 42,
    #             add_key_loss=False,
    #             add_spe_loss=False,
    #             random_non_queries= True ,
    #             vocab_size=VOCAB_SIZE
    #         )
    #     all_data.extend(data)
    # data_path = f"/nvme1/zecheng/data/MQAR/for_tiny/mqar-v0-standard/{split}.jsonl"
    # print("example_data: \n", all_data[0])
    # print("example_data: \n", all_data[len(all_data)//2])
    # print("example_data: \n", all_data[-1])
    # auto_save_data(all_data, data_path)

    # all_data = []
    # split = "test"
    # key_length = 2
    # value_length = 2
    # for input_len  in [64,256,512]:
    #     for num_kv_pairs in [1, 4, 8 ,16 , 32]:
    #         if (num_kv_pairs * (3+value_length+key_length)) >= input_len//2 : continue
    #         data = MQARDataset.build_v6_standard_robustness_dataset(
    #                 input_seq_len=input_len,
    #                 num_kv_pairs=num_kv_pairs,
    #                 num_examples=100,
    #                 key_length=key_length,
    #                 value_length=value_length,
    #                 context_len = None,
    #                 mask_value = True ,
    #                 random_seed = 4567,
    #                 add_key_loss=False,
    #                 add_spe_loss=False,
    #                 random_non_queries= True ,
    #                 vocab_size=20000
    #             )
    #         all_data.extend(data)
    # data_path = f"/nvme1/zecheng/data/MQAR/mqar-v6-k2v2-robustness/{split}.jsonl"
    # print("example_data: \n", all_data[0])
    # print("example_data: \n", all_data[len(all_data)//2])
    # print("example_data: \n", all_data[-1])
    # auto_save_data(all_data, data_path)
    
    # VOCAB_SIZE=20000
    # # # max_seq_len = 512
    # v = "v0"
    # key_len=1
    # value_len=1
    # split = ["train", "valid", "test"]
    # max_seq_len = ["64K", 512, 66000]
    # comment="64k-pair1"
    # for idx in range(2,3):
    #     configs = generate_configs(num_examples=100, max_seq_len=max_seq_len[idx], split=split[idx], dataset_version=v, key_len=key_len, value_len=value_len, VOCAB_SIZE=20000)
    #     build_dataset(configs=configs, split=split[idx], dataset_version=v, version_name=f"k{str(key_len)}v{str(value_len)}", data_name=str(max_seq_len[0]), comment=comment)
    

    VOCAB_SIZE = 20000
    all_data = []
    test_input_kv=[(64,1),(64,2),(64,4),
                   (128,1),(128,2),(128,4),(128,8),
                   (256,1),(256,2),(256,4),(256,8),(256,16),
                   (512,1),(512,2),(512,4),(512,8),(512,16), (512,32)]
    split = "test"
    for input_l, kv in test_input_kv:
        data = MQARDataset.build_robustness_split_mqar(
                input_seq_len=input_l,
                num_kv_pairs=kv,
                num_examples=100,
                key_length=1,
                value_length=1,
                context_len = None,
                mask_value = False ,
                random_seed = 4567,
                add_key_loss=False,
                add_spe_loss=False,
                random_non_queries= False ,
                vocab_size=VOCAB_SIZE
            )
        all_data.extend(data)
    data_path = f"/nvme/ywj/data/robustness_v0_position/{split}.jsonl"
    print("example_data: \n", all_data[0])
    print("example_data: \n", all_data[len(all_data)//2])
    print("example_data: \n", all_data[-1])
    auto_save_data(all_data, data_path)