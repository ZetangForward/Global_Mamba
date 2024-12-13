# Load model directly
import torch
import tqdm
from transformers import AutoModel,AutoTokenizer,MambaForCausalLM
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from custom_dataset.pg19 import PG19Data
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
# model = AutoModel.from_pretrained("state-spaces/mamba-2.8b")
device = "cuda:7"
model = MambaForCausalLM.from_pretrained("/nvme/hf_models/mamba-130m-hf").to(device)
tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/mamba-130m-hf")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
content = load_dataset("/nvme1/zecheng/DeciMamba/hf_cache/data")['test']
test_dataset = PG19Data(content=content, tokenizer=tokenizer, split="test")
data = DataLoader(test_dataset, 
                  batch_size=1, 
                  num_workers=1, \
                  pin_memory=True, 
                  drop_last=False, 
                  shuffle=False)

def evaluate_validation_set_ppl_test(model, model_processor, data_loader_val, epoch=0, cur_step=1, num_samples_to_log=None):
    minimal_stride = 1
    max_amount_of_windows = 1
    ce_loss = CrossEntropyLoss()
    # import pdb;pdb.set_trace()
    dataset_val = data_loader_val

    context_lengths = [i * 1000 for i in range (1, 64, 4)] #, 1024, 2048, 4096, 8192, 16384, 32768]
    ppl_per_context_length = {length: [] for length in context_lengths}

    for i, sample in enumerate(tqdm(dataset_val)):
        if i>=10: break
        seq_len = sample['input_ids'].size(1)
        print(f'Processing sample {i}, seq_len = {seq_len//1000}K')

        for window_size in context_lengths:
            if seq_len < window_size:
                print(f'Skipping context length {window_size}, seq_len = {seq_len//1000}K < window_size = {window_size//1000}K')
                continue

            nlls = []
            stride = (seq_len - window_size) // max_amount_of_windows
            stride = max(stride, minimal_stride)
            trg_len = window_size

            for begin_loc in range(0, seq_len - window_size + 1, stride):
                end_loc = begin_loc + window_size
                input_ids = sample['input_ids'][:, begin_loc:end_loc].to(model.device)
                target_ids = input_ids.clone()

                with torch.no_grad():
                    target_ids = target_ids[:, -trg_len:]
                    # import pdb;pdb.set_trace()
                    outputs = model(input_ids)

                    logits = outputs.logits
                    logits = logits.view(-1, logits.size(-1))
                    # params = outputs.record_params
                    # import pdb;pdb.set_trace()
                    target_ids = target_ids.view(-1)
                    neg_log_likelihood = ce_loss(logits.squeeze()[:-1], target_ids.squeeze()[1:])

                nlls.append(neg_log_likelihood.item())

            if nlls:
                ppl = torch.exp(torch.tensor(nlls).mean()).cpu().float()
            else:
                ppl = torch.tensor(float('inf'))

            print(f'Calculated perplexity for context length {window_size}: {ppl:.2f}')
            ppl_per_context_length[window_size].append(ppl.item())

    avg_ppl_per_context_length = {length: np.mean(ppls) if ppls else float('inf')
                                   for length, ppls in ppl_per_context_length.items()}
    # import pdb;pdb.set_trace()

    val_log = {
        'score': np.mean(list(avg_ppl_per_context_length.values())),
        'ppl_per_ctx_len': {
            'epoch': epoch,
            'step': cur_step,
            'ppl_per_context_length': '\t'.join(f'{avg_ppl_per_context_length[length]:.2f}' for length in context_lengths)
        }
    }

    print(tabulate([['score:'] + [f'{avg_ppl_per_context_length[length]:.2f}' for length in context_lengths]],
                   headers=['ctx len:'] + [f'{length}' for length in context_lengths],
                   tablefmt='pretty')) 

    return [], val_log


evaluate_validation_set_ppl_test(model,tokenizer,data)