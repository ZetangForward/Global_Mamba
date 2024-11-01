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


    def build_position_mqar(input_seq_len, num_kv_pairs, \
                        vocab_size=20000, insert_noise=True, \
                        key_length=1, value_length=1,\
                        split="train", position_type="standard", random_seed=42):
        # import pdb;pdb.set_trace()
        if split == "train":
            num_examples = 100
        elif split == "valid":
            num_examples = 100
        elif split == "test":
            num_examples = 100

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        single_kv_pair_lens =  2

        if position_type == "standard":
            context_size = num_kv_pairs * 2
        else:
            context_size = input_seq_len // 2

        key_vocab_size = vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)
        if position_type == "shuffle":
            keys = np.random.choice(key_choices, size=(num_examples, num_kv_pairs, 1), replace=True)
            values = np.random.choice(value_choices, size=(num_examples, num_kv_pairs, 1), replace=True)
        else:
            keys_unshuffled = np.tile(key_choices, (num_examples, 1))
            keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

            values_unshuffled = np.tile(value_choices, (num_examples, 1))
            values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)



        if position_type == "standard":
            kvs = np.zeros((num_examples, context_size), dtype=np.int64)
            kvs[:, 0::2] = keys
            kvs[:, 1::2] = values
        elif position_type == "last":
            kvs = np.zeros((num_examples, input_seq_len // 2 ), dtype=np.int64)
            kvs[:, (context_size-num_kv_pairs*2)::2] = keys
            kvs[:, (context_size-num_kv_pairs*2+1)::2] = values
        elif position_type == "shuffle":
            kvs = np.zeros((num_examples, input_seq_len // 2 ), dtype=np.int64)
            kv_space = input_seq_len // 2  // single_kv_pair_lens 
            kv_X = np.stack([np.arange(kv_space, dtype=int)] * num_examples)
            kv_gaps = np.apply_along_axis(np.random.choice, axis=1, arr=kv_X, replace=False, size=num_kv_pairs)
            kv_start_indices = kv_gaps * (single_kv_pair_lens)
            expanded_start_indices = kv_start_indices[:,:,None]  + np.arange(1) 
            expanded_start_indices_1 = kv_start_indices[:,:,None] + np.arange(1) + 1
            kvs[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys
            kvs[np.arange(num_examples)[:, None, None], expanded_start_indices_1] = values


        space = (input_seq_len - context_size) // 2
        p = 0.01 * np.arange(1, space + 1) ** (0.01 - 1)
        p = p / p.sum()

        # import pdb;pdb.set_trace()
        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
        queries = np.zeros((num_examples, input_seq_len - context_size), dtype=np.int64)
        
        if position_type == "shuffle":
            start_indices = gaps * (single_kv_pair_lens)
            expanded_start_indices = start_indices[:,:,None] + np.arange(key_length)  # 添加偏移量
            queries[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys
            examples = np.concatenate([kvs, queries], axis=1)
            labels = np.full((num_examples, input_seq_len ), -100, dtype=np.int64)
            value_start_index = start_indices + context_size + key_length
            value_expanded_start_indices = value_start_index[:,:,None] + np.arange(value_length)  # 添加偏移量
            labels[np.arange(num_examples)[:, None, None], value_expanded_start_indices] = values

        else:
            np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
            examples = np.concatenate([
                kvs, 
                queries
            ], axis=1)
            labels = np.full((num_examples, input_seq_len ), -100, dtype=np.int64)
            np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

        
      
        inputs, labels = torch.tensor(examples[:, :]), torch.tensor(labels[:, :])

        if insert_noise:
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
            all_test_data.append(data_dict) 
        return all_test_data

    def build_ngram_mqar(input_seq_len, num_kv_pairs, \
                        vocab_size=20000, insert_noise=True, \
                        key_length=1, value_length=1,\
                        split="train", position_type="standard",random_seed=42):
        if split == "train":
            num_examples = 100
        elif split == "valid":
            num_examples = 100
        elif split == "test":
            num_examples = 100

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        single_kv_pair_lens = (3 + key_length + value_length)
        if num_kv_pairs * single_kv_pair_lens > input_seq_len//2:
            return None
        if position_type == "standard":
            context_size = num_kv_pairs * single_kv_pair_lens
        elif position_type == "shuffle":
            context_size = input_seq_len//2

 
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(20, key_vocab_size + 20)
        value_choices = np.arange(key_vocab_size + 20, vocab_size + 20)

        keys = np.random.choice(key_choices, size=(num_examples, num_kv_pairs, key_length), replace=True)
        values = np.random.choice(value_choices, size=(num_examples, num_kv_pairs, value_length), replace=True)

        kvs = np.zeros((num_examples, context_size), dtype=np.int64)
        if position_type == "standard":
            for i in range(num_kv_pairs):
                kvs[:, single_kv_pair_lens * i] = 10  # <s1>
                kvs[:, single_kv_pair_lens * i + 1:single_kv_pair_lens * i + 1 + key_length] = keys[:, i, :]
                kvs[:, single_kv_pair_lens * i + 1 + key_length] = 11  # <s2>
                kvs[:, single_kv_pair_lens * i + 2 + key_length:single_kv_pair_lens * i + 2 + key_length + value_length] = values[:, i, :]
                kvs[:, single_kv_pair_lens * i + 2 + key_length + value_length] = 10  # <s1>
        else:
            kv_space = input_seq_len // 2  // single_kv_pair_lens 
            kv_X = np.stack([np.arange(kv_space, dtype=int)] * num_examples)
            kv_gaps = np.apply_along_axis(np.random.choice, axis=1, arr=kv_X, replace=False, size=num_kv_pairs)
            kv_start_indices = kv_gaps * (single_kv_pair_lens)

            kvs[np.arange(num_examples)[:, None], kv_start_indices] = 10
            kvs[np.arange(num_examples)[:, None], kv_start_indices + 1 + key_length + value_length + 1] = 10
            kvs[np.arange(num_examples)[:, None], kv_start_indices + 1 + key_length] = 11
            expanded_start_indices = kv_start_indices[:,:,None] + 1 + np.arange(key_length)  
            expanded_start_indices_1 = kv_start_indices[:,:,None] + 2 + key_length +  np.arange(value_length) 
            kvs[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys
            kvs[np.arange(num_examples)[:, None, None], expanded_start_indices_1] = values

        space = (input_seq_len - context_size) // single_kv_pair_lens  # Adjust for longer keys in queries
        p = 0.01 * np.arange(1, space + 1) ** (0.01 - 1)
        p = p / p.sum()


        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
        queries = np.zeros((num_examples, input_seq_len - context_size ), dtype=np.int64)

        start_indices = gaps * (single_kv_pair_lens)
        queries[np.arange(num_examples)[:, None], start_indices] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length + value_length + 1] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length] = 11
        expanded_start_indices = start_indices[:,:,None] + 1 + np.arange(key_length)  
        queries[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys

        examples = np.concatenate([kvs, queries], axis=1)

        labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)

        value_start_index = start_indices + context_size + 2 + key_length
        value_expanded_start_indices = value_start_index[:,:,None] + np.arange(value_length)  
        labels[np.arange(num_examples)[:, None, None], value_expanded_start_indices] = values

        inputs = torch.tensor(examples[:, :])

        # mask_value:
        mask = torch.Tensor(labels[:,:-1] != -100).bool()
        inputs[mask] = 12

       
        # add_spe_loss:
        if split=="train":
            key_start_index = start_indices + context_size
            labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length + value_length + 1] = 10
        labels = torch.tensor(labels[:, :])

        if insert_noise:
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
            all_test_data.append(data_dict) 

        return all_test_data
    
    def build_ngram_robustness_mqar(input_seq_len, num_kv_pairs, \
                        vocab_size=20000, insert_noise=True, \
                        key_length=1, value_length=1,\
                        split="train", position_type="standard",random_seed=42):
        if split == "train":
            num_examples = 100
        elif split == "valid":
            num_examples = 100
        elif split == "test":
            num_examples = 100

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        single_kv_pair_lens = (3 + key_length + value_length)
        if num_kv_pairs * single_kv_pair_lens > input_seq_len//2:
            return None
        if position_type == "standard":
            context_size = num_kv_pairs * single_kv_pair_lens
        elif position_type == "shuffle":
            context_size = input_seq_len//2

 
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(20, key_vocab_size + 20)
        value_choices = np.arange(key_vocab_size + 20, vocab_size + 20)

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

        kvs = np.zeros((num_examples, kvpairs_size), dtype=np.int64)
        for i in range(num_kv_pairs):
            kvs[:, single_kv_pair_lens * i] = 10  # <s1>
            kvs[:, single_kv_pair_lens * i + 1:single_kv_pair_lens * i + 1 + key_length] = keys[:, i, :]
            kvs[:, single_kv_pair_lens * i + 1 + key_length] = 11  # <s2>
            kvs[:, single_kv_pair_lens * i + 2 + key_length:single_kv_pair_lens * i + 2 + key_length + value_length] = values[:, i, :]
            kvs[:, single_kv_pair_lens * i + 2 + key_length + value_length] = 10  # <s1>

        # import pdb;pdb.set_trace()
        space = (input_seq_len - kvpairs_size) // single_kv_pair_lens 
        p = 0.01 * np.arange(1, space + 1) ** (0.01 - 1)
        p = p / p.sum()


        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=None, size=num_kv_pairs)
        queries = np.zeros((num_examples, input_seq_len - kvpairs_size ), dtype=np.int64)

        start_indices = gaps * (single_kv_pair_lens)
        queries[np.arange(num_examples)[:, None], start_indices] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length + value_length + 1] = 10
        queries[np.arange(num_examples)[:, None], start_indices + 1 + key_length] = 11
        expanded_start_indices = start_indices[:,:,None] + 1 + np.arange(key_length) 
        queries[np.arange(num_examples)[:, None, None], expanded_start_indices] = keys

        examples = np.concatenate([kvs, queries], axis=1)

        labels = np.full((num_examples, input_seq_len ), -100, dtype=np.int64)

        value_start_index = start_indices + kvpairs_size + 2 + key_length
        value_expanded_start_indices = value_start_index[:,:,None] + np.arange(value_length)  
        labels[np.arange(num_examples)[:, None, None], value_expanded_start_indices] = values

        inputs = torch.tensor(examples[:, :])
        mask = torch.Tensor(labels[:,:] != -100).bool()
        inputs[mask] = 12

        if split=="train":
            key_start_index = start_indices + context_size
            labels[np.arange(num_examples)[:, None], key_start_index + 1 + key_length + value_length + 1] = 10
            

        labels = torch.tensor(labels[:, :])
        if insert_noise:
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


if __name__ == '__main__':
    VOCAB_SIZE = 20000
    for split in ["train", "valid", "test"]:
        for position_type in ["standard", "last", "shuffle"]:
            all_data = []
            test_input_kv=[(64,1),(64,2),(64,4),(64,8),(64,16),
                        (128,1),(128,2),(128,4),(128,8),(128,16),(128,32),
                        (256,1),(256,2),(256,4),(256,8),(256,16),(256,32),(256,64),
                        (512,1),(512,2),(512,4),(512,8),(512,16), (512,32),(512,64), (512,128)]
            seed = {"train":42, "valid":1234, "test":4567}
            for idx, (input_l, kv) in enumerate(test_input_kv):
                data = MQARDataset.build_position_mqar(
                        input_seq_len=input_l,
                        num_kv_pairs=kv,
                        vocab_size=VOCAB_SIZE,
                        insert_noise=True,
                        key_length=1,
                        value_length=1,
                        split=split,
                        position_type=position_type,
                        random_seed = seed[split]+idx,
                    )
                all_data.extend(data)
            data_path = f"data/position-{position_type}/{split}.jsonl"
            auto_save_data(all_data, data_path)

    for split in ["train", "valid", "test"]:
        for key_length,value_length in [(1,1), (1,2), (2,2)]:
            all_data = []
            test_input_kv=[(64,1),(64,2),(64,4),(64,8),(64,16),
                        (128,1),(128,2),(128,4),(128,8),(128,16),(128,32),
                        (256,1),(256,2),(256,4),(256,8),(256,16),(256,32),(256,64),
                        (512,1),(512,2),(512,4),(512,8),(512,16), (512,32),(512,64), (512,128)]
            position_type = "standard"
            seed = {"train":42, "valid":1234, "test":4567}
            for idx, (input_l, kv) in enumerate(test_input_kv):
                data = MQARDataset.build_ngram_mqar(
                        input_seq_len=input_l,
                        num_kv_pairs=kv,
                        vocab_size=VOCAB_SIZE,
                        insert_noise=True,
                        key_length=key_length,
                        value_length=value_length,
                        split=split,
                        position_type=position_type,
                        random_seed = seed[split]+idx,
                    )
                if data:
                    all_data.extend(data) 
            data_path = f"data/ngram-k{key_length}v{value_length}-{position_type}/{split}.jsonl"
            auto_save_data(all_data, data_path)

    all_data = []
    test_input_kv=[(64,1),(64,4),(64,8),
                   (256,1),(256,4),(256,8),(256,16),
                   (512,1),(512,4),(512,8),(512,16), (512,32)]
    split = "test"
    position_type = "standard"
    seed = {"train":42, "valid":1234, "test":4567}
    key_length = 2
    value_length = 2
    for idx, (input_l, kv) in enumerate(test_input_kv):
        data = MQARDataset.build_ngram_robustness_mqar(
                input_seq_len=input_l,
                num_kv_pairs=kv,
                vocab_size=VOCAB_SIZE,
                insert_noise=True,
                key_length=key_length,
                value_length=value_length,
                split=split,
                position_type=position_type,
                random_seed = seed[split]+idx,
            )
        if data:
            all_data.extend(data) 
    data_path = f"data/robustness-k{key_length}v{value_length}/{split}.jsonl"
    auto_save_data(all_data, data_path)