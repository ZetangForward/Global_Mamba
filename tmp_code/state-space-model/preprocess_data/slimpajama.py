import multiprocessing
from modelzipper.tutils import *
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm

max_seq_length = 6000

raw_data = load_from_disk("/nvme/zecheng/data/slimpajama-per-source-length-upsample-gpt-hf")
tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/mamba-1.4b-hf")

def process_data(item):
    sample = item['input_ids']  # list format
    split_sample = [sample[i:i+max_seq_length] for i in range(0, len(sample), max_seq_length)]
    if len(split_sample[-1]) < max_seq_length * 0.8:
        split_sample.pop()  # drop last one if it's too short
    str_sample = [tokenizer.decode(item) for item in split_sample]
    return str_sample

if __name__ == "__main__":
    with multiprocessing.Pool(48) as pool:
        processed_data = list(tqdm(pool.imap(process_data, raw_data), total=len(raw_data)))

    auto_save_data(processed_data, "/nvme/zecheng/data/simpajama-processed/processed/processed_up_sample_6k.jsonl")
