from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import torch
import glob


class SynPasskeySearchDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs["max_seq_length"]
        self.cluster_batch = kwargs["cluster_batch"]
        self.filter_length(kwargs["testing_max_ctx"])

    def filter_length(self, max_ctx_length=12000, sort=False):
        new_content = []
        print_c(f"begin to filter the context length | total {len(self.content)} instances", "yellow")
        for item in self.content:
            if item['before_insert_context_length'] <= max_ctx_length:
                new_content.append(item)
        if sort:
            new_content = sorted(new_content, key=lambda x: x['ctx_length'], reverse=True)  # from long to short
        self.content = new_content
        print_c(f"filtering finished | total {len(self.content)} instances", "yellow")


    @classmethod
    def load_context(cls, fpath, ctx_len=10000, tokenizer=None):
        context = ""
        for file in glob.glob(fpath):
            with open(file, 'r') as f: 
                context += f.read()
        context = context * 3  # prevent the context from being too short
        tokenized_context = tokenizer(context, return_tensors="pt").input_ids[0][:ctx_len]
        # tok_ids_len = len(tokenized_context[0])
        # RATIO = len(context) / tok_ids_len
        # context = context[: int(ctx_len * RATIO)]
        return tokenized_context
    
    
    @classmethod
    def insert_needle_str(cls, context, needle, depth):
        context = context.split(".")
        c_len = len(context)
        needle_place = int(depth * c_len)
        context = ".".join(context[:needle_place]) + " ." + needle + ". ".join(context[needle_place:])
        return context


    @classmethod
    def insert_needle_token_ids(cls, context, needle, depth):
        needle_place = int(depth * len(context))
        bos_pos, eos_pos = needle_place, needle_place + len(needle)
        context = torch.cat((context[:needle_place], needle, context[needle_place:]))
        return context, bos_pos, eos_pos

    def insert_kv_pairs(context, kv_pairs, insert_points):
        """
        在指定位置插入kv_pairs到context中。
        """
        context = torch.tensor(context)
        kv_pairs = torch.tensor(kv_pairs)
        batch_size, context_len = context.shape
        _, kv_pairs_len = kv_pairs.shape
        new_context_len = context_len + kv_pairs_len  # 插入kv_pairs后的长度
        new_context = torch.zeros((batch_size, new_context_len), dtype=context.dtype)

        for i in range(batch_size):
            insert_point = insert_points[i]
            # 在插入点之前的部分
            new_context[i, :insert_point] = context[i, :insert_point]
            # 插入kv_pairs
            new_context[i, insert_point:insert_point+kv_pairs_len] = kv_pairs[i]
            # 在插入点之后的部分
            new_context[i, insert_point+kv_pairs_len:] = context[i, insert_point:context_len]
        
        return new_context


    @classmethod
    def build_dataset(cls, vocab_size, number, context_len, key_len=8, value_len=40, random_seed=42, insert_into_context=False):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        key_choices = np.arange(20, vocab_size)
        value_choices = np.arange(20, vocab_size)

        keys_unshuffled = np.tile(key_choices, (number, 1))
        keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=key_len)
        values_unshuffled = np.tile(value_choices, (number, 1))
        values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=value_len)
        keys[:,0] = 10      # 10 作key的标识符
        keys[:,-1] = 10
        values[:,0] = 11    # 11 作value的标识符
        values[:,-1] = 11
        
        kv_pairs = np.concatenate([keys, values], axis=1)
        
        context = np.arange(20 , vocab_size+20)
        context = np.tile(context, (number, 1))
        context = np.apply_along_axis(np.random.choice, 1, context, replace=False, size=context_len)
        
        if insert_into_context:
            shuffled_indices  = np.random.permutation(kv_pairs.shape[0])
            shuffled_kv_pairs = kv_pairs[shuffled_indices]
            
            _, replacement_len = shuffled_kv_pairs.shape

            # 假设 replace_points 是每行替换开始的位置
            replace_points = torch.randint(0, context_len - replacement_len + 1, (number,))

            context_updated = context.copy()
            for i in range(number):
                start = replace_points[i].item()  # 使用.item()获取整数值
                end = start + replacement_len
                context_updated[i, start:end] = shuffled_kv_pairs[i]

            insert_points_pre = [torch.randint(0, max(1,start), (1,)).item() for start in replace_points]
            insert_points_post = []
            insert_point = [ 0 for start in replace_points]
            for end in replace_points + replacement_len:
                if end < context_len:
                    cur_post = torch.randint(end, context_len, (1,)).item()
                    insert_points_post.append(cur_post)
                else:
                    insert_points_post.append(context_len - 1)
            for i in range(number):
                if torch.rand(1) < 0.5:
                    insert_point[i] = insert_points_pre[i]
                else:
                    insert_point[i] = insert_points_post[i]
                    
        else:
            insert_point = torch.randint(0, context_len , (number,))
            context_updated = context.copy()

        context_updated = cls.insert_kv_pairs(context_updated, kv_pairs, insert_point)
        inputs = torch.cat([context_updated, torch.tensor(keys), torch.full((number,1),11)],dim=1)


        all_test_data = []
        for i in range(inputs.size(0)):  
            bos_pos = int(insert_point[i])
            key = keys[i].tolist()
            value = values[i][1:].tolist()
            input_ids = inputs[i].tolist()
            all_test_data.append(
                {
                    "bos_pos": bos_pos,
                    "key": key,
                    "value": value,
                    "ctx_len": context_len,
                    "input_ids": input_ids
                }
            )
            # import pdb;pdb.set_trace()
        return all_test_data




    def cluster_batch_fn(self):
        tmp = [item['source'] + ' ' + item['target'] for item in self.content]
        # tok_tmp = [self.tokenizer(item, return_tensors="pt") for item in tmp]
        sorted_tok_tmp = sorted(tmp, key=lambda x: len(x.split()))
        self.content = sorted_tok_tmp

    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        item = self.content[index]
        item['input_ids'] = item.pop('context_ids')
        return item
    
        passkey_context = item.pop('context_ids')
        tokenized_sequence = self.tokenizer(  # re-tokenize to get attention mask
            passkey_context,  
            return_tensors="pt",
        )

        input_ids = tokenized_sequence.input_ids[0]
        attention_mask = tokenized_sequence.attention_mask[0]
        real_length = attention_mask.size(-1)
        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'real_length': real_length,
        }

        res.update(item)

        return res



if __name__ == '__main__':
    x = SynPasskeySearchDataset.build_dataset(vocab_size=20000,number=200,context_len=512,random_seed=42,insert_into_context=False)
    all_train_data = []
    all_test_data = []
    VOCAB_SIZE = 20000
    train_configs = [
        {"vocab_size": VOCAB_SIZE, "context_len": 512, "num_examples": 5000},
        {"vocab_size": VOCAB_SIZE, "context_len": 1024, "num_examples": 5000},
        {"vocab_size": VOCAB_SIZE, "context_len": 2048, "num_examples": 5000},
        {"vocab_size": VOCAB_SIZE, "context_len": 4096, "num_examples": 5000},
        {"vocab_size": VOCAB_SIZE, "context_len": 8192, "num_examples": 5000},
        {"vocab_size": VOCAB_SIZE, "context_len": 16384, "num_examples": 5000},
    ]

   
    
    for train_config in train_configs:
        vocab_size = train_config["vocab_size"]
        context_len =  train_config["context_len"]
        num_examples = train_config["num_examples"]
        train_data = SynPasskeySearchDataset.build_dataset(
            vocab_size=vocab_size,
            number=num_examples,
            context_len=context_len,
            random_seed=42,
            insert_into_context=False)
        train_data_path = "/public/home/ljt/tzc/data/SynPasskeySearch/synpasskey_16k_no_context.jsonl"
        with open(train_data_path,"a+") as f:
            for item in train_data:
                json.dump(item, f)
                f.write("\n")
    
    for train_config in train_configs:
        vocab_size = train_config["vocab_size"]
        context_len =  train_config["context_len"]
        num_examples = train_config["num_examples"]
        train_data = SynPasskeySearchDataset.build_dataset(
            vocab_size=vocab_size,
            number=num_examples,
            context_len=context_len,
            random_seed=42,
            insert_into_context=True)
        train_data_path = "/public/home/ljt/tzc/data/SynPasskeySearch/synpasskey_16k_in_context.jsonl"
        with open(train_data_path,"a+") as f:
            for item in train_data:
                json.dump(item, f)
                f.write("\n")
