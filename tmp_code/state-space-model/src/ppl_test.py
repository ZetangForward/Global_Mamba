import os
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from tabulate import tabulate

def evaluate_validation_set_ppl_test(model, model_processor, data_loader_val, config, epoch, cur_step, num_samples_to_log=None):
    minimal_stride = 1
    max_amount_of_windows = 1
    ce_loss = CrossEntropyLoss()

    dataset_val = data_loader_val.predict_dataloader()
    context_lengths = [2048,4096,6144,8192,16384,32768,64000,12800,25600,51200,10240] #, 1024, 2048, 4096, 8192, 16384, 32768]
    ppl_per_context_length = {length: [] for length in context_lengths}
    # import pdb;pdb.set_trace()

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
            # trg_len = window_size
            trg_len = 128

            for begin_loc in range(0, seq_len - window_size + 1, stride):
                end_loc = begin_loc + window_size
                input_ids = sample['input_ids'][:, begin_loc:end_loc].to(model.device)
                target_ids = input_ids.clone()

                with torch.no_grad():
                    target_ids = target_ids[:, -trg_len:]
                    outputs = model(input_ids)
                    logits = outputs.logits
                    logits = logits.view(-1, logits.size(-1))
                    params = outputs.record_params
                    save_params(params, window_size)
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


def save_params(params, window_size):
    analysis_root_path = f"/nvme1/zecheng/analysis/ppl_pg19/longconv-512-7_15_23/context_{window_size}/"
    os.makedirs(analysis_root_path, exist_ok=True)
    analysis_path = os.path.join(analysis_root_path, "params.pt")
    torch.save([params], analysis_path)