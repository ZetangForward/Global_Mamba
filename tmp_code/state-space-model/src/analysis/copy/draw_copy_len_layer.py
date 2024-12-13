import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from modelzipper.tutils import *
import matplotlib.pyplot as plt
import pickle



def get_predictions(fpath):
    predictions = auto_read_data(fpath)
    sentence_level = 0
    token_level = 0
    total_tokens = 0
    for item in predictions:
        pred = item['predictions'].squeeze(0)
        label = item['labels'].squeeze(0)
        pred = pred[:label.shape[-1]]
        sentence_level += (((label == pred).sum()) == label.shape[-1]).item()
        token_level += ((label == pred).sum()).item()
        total_tokens += label.shape[-1]
    return sentence_level / len(predictions), token_level / total_tokens

if __name__ == '__main__':
    results = {}
    for model in ["mamba-370m-s4", "mamba-370m-s8", "mamba-370m-s16", "mamba-370m-s32", "mamba-370m-s64"]:
        root_path = "/public/home/ljt/tzc/evaluation/Copy"
        # len_list = [10, 64, 128 ]
        # len_list.extend([i for i in range(256, 576, 32)])
        len_list=[i for i in range(4,512,4)]
        sentence_scores = []
        token_scores = []
        for len_ in len_list:
            pred_path = root_path + f"/{model}/Copy-{model}-len{len_}/predictions.pkl"
            if not os.path.exists(pred_path):   
                continue
            try:
                sentence_score, token_score = get_predictions(pred_path)
            except: 
                sentence_score, token_score = 0, 0
            sentence_scores.append(sentence_score)
            token_scores.append(token_score)
        results[model] = {'sentence': sentence_scores, 'token': token_scores}

    # 绘制图表
    for model, scores in results.items():
        plt.figure(figsize=(10, 5))
        plt.plot(len_list, scores['sentence'], label='Sentence Level')
        plt.title(f'Sentence Level Scores for {model}')
        plt.xlabel('Sequence Length')
        plt.ylabel('Score')
        plt.legend()
        save_path = "/public/home/ljt/tzc/evaluation/analysis/Copy/"+f"{model}_Sentence_Level.png"
        plt.savefig(save_path)

        plt.figure(figsize=(10, 5))
        plt.plot(len_list, scores['token'], label='Token Level')
        plt.title(f'Token Level Scores for {model}')
        plt.xlabel('Sequence Length')
        plt.ylabel('Score')
        plt.legend()
        save_path = "/public/home/ljt/tzc/evaluation/analysis/Copy/"+f"{model}_Token_Level.png"
        plt.savefig(save_path)
