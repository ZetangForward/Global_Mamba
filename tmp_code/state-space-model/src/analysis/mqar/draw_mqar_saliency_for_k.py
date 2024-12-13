import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from modelzipper.tutils import *

avg_a = 1

def draw_len_layer(data_path, score_metrics, model):
    data = auto_read_data(data_path)
    scores_dict = {}

    for d in data:
        input_seq_len = int(d['input_seq_len'])
        kv_pairs = int(d['kv_pairs'])
        # import pdb;pdb.set_trace()
        scores = d['saliency']
        # score_list = []
        # for score in scores:
        #     score_list.append(float(score[score_metrics]))
            
       
        scores = [float(item[score_metrics][0].cpu().item() if isinstance(item[score_metrics],tuple) else item[score_metrics].cpu().item()) for item in d['saliency']]
        avg_scores = [sum(scores[i:i+avg_a]) / avg_a for i in range(0, len(scores), avg_a)]
    
        if input_seq_len not in scores_dict:
            scores_dict[input_seq_len] = []

        scores_dict[input_seq_len].append(avg_scores)
        

    avg_scores_dict = {k: [sum(col) / len(col) for col in zip(*v)] for k, v in scores_dict.items()}
    min_val = min([min(values) for values in avg_scores_dict.values()])
    max_val = max([max(values) for values in avg_scores_dict.values()])
    avg_scores_dict = {key: [(val - min_val) / (max_val - min_val) for val in values] for key, values in avg_scores_dict.items()}

    df = pd.DataFrame(avg_scores_dict)
    
    
    sns.heatmap(df, cmap='BuGn')
    save_path = f'/public/home/ljt/tzc/evaluation/analysis/len-layer/{model}/{score_metrics}_heatmap.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()


def draw_len_kv(data_path, score_metrics, model):
    data = auto_read_data(data_path)
    scores_dict = {}

    for d in data:
        # 直接将 input_seq_len 和 kv_pairs 转换为整数
        input_seq_len = int(d['input_seq_len'])
        kv_pairs = int(d['kv_pairs'])
      
        scores = [float(item[score_metrics][0].cpu().item() if isinstance(item[score_metrics],tuple) else item[score_metrics].cpu().item()) for item in d['saliency']]

        avg_scores = [sum(scores[i:i+avg_a]) / avg_a for i in range(0, len(scores), avg_a)]
        
        key = (input_seq_len, kv_pairs)
        if key not in scores_dict:
            scores_dict[key] = []

        scores_dict[key].extend(avg_scores)
    
    
    # import pdb;pdb.set_trace()
    
    # all_scores = [score for scores in scores_dict.values() for score in scores]
    
    # min_score = min(all_scores)
    # max_score = max(all_scores)

    # # 对每个数值进行归一化处理
    # normalized_p_scores_dict = {
    #     key: [(score - min_score) / (max_score - min_score) for score in values]
    #     for key, values in scores_dict.items()
    # }

    
    # 首先将键分开，然后分别转换为整型，并按照数值排序
    input_seq_lens = sorted(set([k[0] for k in scores_dict.keys()]))
    kv_pairs_list = sorted(set([k[1] for k in scores_dict.keys()]))
    
    
    df = pd.DataFrame(index=input_seq_lens, columns=kv_pairs_list)

    for (input_seq_len, kv_pairs), scores in scores_dict.items():
        if np.isnan(np.mean(scores)):
            avg_score = None
        else:
            avg_score = np.mean(scores)
        df.at[input_seq_len, kv_pairs] = avg_score
        
    min_score = df.min().min()
    max_score = df.max().max()

    # 对DataFrame中的值进行归一化处理
    df = (df - min_score) / (max_score - min_score)


    # 为确保正确处理数值类型，再次转换 DataFrame 的索引和列为数值类型
    df.index = pd.to_numeric(df.index)
    df.columns = pd.to_numeric(df.columns)
    
    # 按数值重新排序索引和列，以确保数值顺序
    df = df.sort_index().sort_index(axis=1)
    df = df.apply(pd.to_numeric, errors='coerce')
    sns.set_style('dark')
    
    # colors = ["#0000FF", "#00FF00"]  # 从蓝色过渡到绿色
    # cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    # sns.heatmap(df, cmap=cmap,vmin=df.mean().mean() - (df.std().std()), vmax=df.mean().mean() + (df.std().std()) )
    
    sns.heatmap(df, cmap='BuGn' )
    save_path = f'/public/home/ljt/tzc/evaluation/analysis/len-kv/{model}/{score_metrics}_heatmap.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()

if __name__ == '__main__':
    for model in ["tiny_mamba-s4-tuned", "tiny_mamba-s8-tuned", "tiny_mamba-s16-tuned", "tiny_mamba-s32-tuned", "tiny_mamba-s64-tuned"]:
    # for model in ["mamba-370m-hf-tuned", "mamba-370m-k8-tuned", "mamba-370m-k16-tuned", "mamba-370m-k32-tuned", "mamba-370m-k64-tuned", ]:
    # model = "tiny_mamba-k4-tuned"
        data_path=f"/public/home/ljt/tzc/evaluation/analysis/{model}_conv1d_adapter_score.pkl"
        for score in ["all_kv_avg", "all_kv_sum", "key_avg", "key_sum", "value_avg", "value_sum"]:
            draw_len_layer(data_path, score,model)
            draw_len_kv(data_path, score, model)
