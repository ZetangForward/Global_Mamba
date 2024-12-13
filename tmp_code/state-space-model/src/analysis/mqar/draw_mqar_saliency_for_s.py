import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from modelzipper.tutils import auto_read_data

def draw_heatmap(data, model, type):
    if isinstance(data, pd.DataFrame):
        # Convert data to a numpy array for row-wise operations
        data_array = data.values
    else:
        data_array = data
    normalized_data = np.zeros_like(data_array)

    # Normalize each row
    for i in range(data_array.shape[0]):
        min_val = np.min(data_array[i, :])  # Minimum value in the row
        max_val = np.max(data_array[i, :])  # Maximum value in the row
        if max_val != min_val:  # Avoid division by zero
            normalized_data[i, :] = (data_array[i, :] - min_val) / (max_val - min_val)
        else:
            normalized_data[i, :] = 0  # Set row to zero if all values are the same

    # Convert normalized data back to DataFrame to use DataFrame index and columns
    normalized_df = pd.DataFrame(normalized_data, index=data.index, columns=data.columns)

    plt.figure()
    sns.heatmap(normalized_df, cmap="BuGn")
    filename = f"/public/home/ljt/tzc/evaluation/analysis/MQAR/state_size/{model}-{type}.png"
    plt.savefig(filename)

def draw(data_path, model):
    data = auto_read_data(data_path)
    all_saliency_score = []
    for d in data:
        score_dict = {}
        input_seq_len = int(d['input_seq_len'])
        kv_pairs = int(d['kv_pairs'])

        scores = d['saliency']
        important_place = kv_pairs * 2
        import pdb;pdb.set_trace()
        layer_score_list = []
        # 每层的saliency_score
        for layer_score in scores:
            important_saliency = layer_score['sum_score'][:important_place].sum()
            other_saliency = layer_score['sum_score'][important_place:].sum()
            saliency_score = important_saliency / other_saliency
            layer_score_list.append(saliency_score.item())

        # 两层取平均
        sum_tensor = torch.zeros_like(scores[0]['sum_score'])
        for layer_idx, score in enumerate(scores):
            sum_tensor += score['sum_score']
        score_tensor = sum_tensor / len(scores)

        # 每个ssm_state上的
        state_score_list = []
        for stat in range(score_tensor.shape[-1]):
            stat_score = score_tensor[:, stat]
            important_saliency = stat_score[:important_place].sum()
            other_saliency = stat_score[important_place:].sum()
            saliency_score = important_saliency / other_saliency
            state_score_list.append(saliency_score.item())

        important_saliency = score_tensor[:important_place].sum()
        other_saliency = score_tensor[important_place:].sum()
        saliency_score = important_saliency / other_saliency

        score_dict['input_seq_len'] = input_seq_len
        score_dict['kv_pairs'] = kv_pairs
        score_dict['layer_score'] = layer_score_list
        score_dict['stat_score'] = state_score_list
        score_dict['score'] = saliency_score.item()
        all_saliency_score.append(score_dict)

    # Drawing the first heatmap
    df = pd.DataFrame(all_saliency_score)
    pivot_table = df.pivot(index="input_seq_len", columns="kv_pairs", values="score")
    draw_heatmap(pivot_table, model, 'len_kv')
    
    # Drawing layer scores heatmap
    layer_scores = []
    for item in all_saliency_score:
        for idx, score in enumerate(item['layer_score']):
            layer_scores.append({'input_seq_len': item['input_seq_len'], 'index': idx, 'score': score})

    df_layer = pd.DataFrame(layer_scores)
    pivot_layer = df_layer.pivot_table(index="input_seq_len", columns="index", values="score", aggfunc="mean")
    draw_heatmap(pivot_layer, model, 'len_layer')

    # Drawing state scores heatmap
    stat_scores = []
    for item in all_saliency_score:
        for idx, score in enumerate(item['stat_score']):
            stat_scores.append({'input_seq_len': item['input_seq_len'], 'index': idx, 'score': score})

    df_stat = pd.DataFrame(stat_scores)
    pivot_stat = df_stat.pivot_table(index="input_seq_len", columns="index", values="score", aggfunc="mean")
    draw_heatmap(pivot_stat, model, 'len_stat')



if __name__ == '__main__':
    for model in ["tiny_mamba-s4-tuned", "tiny_mamba-s8-tuned", "tiny_mamba-s16-tuned", "tiny_mamba-s32-tuned", "tiny_mamba-s64-tuned"]:
        data_path=f"/public/home/ljt/tzc/evaluation/analysis/MQAR/state_size/{model}_state_adapter_score.pkl"
        draw(data_path, model)

