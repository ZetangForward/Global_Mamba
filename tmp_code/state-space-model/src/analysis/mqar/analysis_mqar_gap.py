import os
import matplotlib.pyplot as plt
import numpy as np
from modelzipper.tutils import *

def get_predictions(input_fpath,prediction_fpath):
    input_data = auto_read_data(input_fpath)
    prediction = auto_read_data(prediction_fpath)
    for idx in range(len(input_data)): 
        input_ids = input_data[idx]['input']
        pred = prediction[idx]['predictions']
        labels = prediction[idx]['labels'][0]
        target_idx = labels!=-100
        pred_value = pred[target_idx]
        label_value = labels[target_idx]
        eval_tensor = pred_value==label_value

        non_neg_labels = [(i, label.item()) for i, label in enumerate(labels) if label != -100]
        gaps = [idx - (input_ids == label).nonzero(as_tuple=True)[0][0].item() for idx, label in non_neg_labels]        
        for tmp in range(len(eval_tensor)):
            gap = gaps[tmp]
            if not gap_score.get(gap):
                gap_score[gap] = {"correct":0 , "total": 0}
            gap_score[gap]["total"] += 1
            gap_score[gap]["correct"] += eval_tensor[tmp]
            

def draw_gap_score(gap_score, save_path):
    gaps = sorted(gap_score.keys())
    accuracies = [gap_score[gap]["correct"] / gap_score[gap]["total"] for gap in gaps]

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(gaps, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy by Gap')
    plt.xlabel('Gap')
    plt.ylabel('Accuracy')
    # plt.grid(True)

    # 设置x轴的标签间隔，这里以每隔100个单位为例
    # xticks = np.arange(min(gaps), max(gaps)+1, 64)
    # plt.xticks(xticks)

    plt.ylim(0, 1)  # y轴的范围设置为0到1

    # 保存图表
    plt.savefig(save_path, bbox_inches='tight')

    # 显示图表
    plt.show()
    

if __name__ == '__main__':
    model_configs = ["tiny_mamba-s4", "tiny_mamba-s8", "tiny_mamba-s16", "tiny_mamba-s32", "tiny_mamba-s64"]
    VOCAB_SIZE=4096
    # test_configs = [
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 1000, "num_kv_pairs": 4},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 1000, "num_kv_pairs": 8},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 64, "num_examples": 1000, "num_kv_pairs": 16},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 128, "num_examples": 1000, "num_kv_pairs": 32},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 256, "num_examples": 1000, "num_kv_pairs": 64},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 512, "num_examples": 1000, "num_kv_pairs": 128},
    #     {"vocab_size": VOCAB_SIZE, "input_seq_len": 1024, "num_examples": 1000, "num_kv_pairs": 256},
    # ]
   
    root_save_dir = "/public/home/ljt/tzc/evaluation/analysis/MQAR/gap/"
    for model_config in model_configs:
        print(model_config)
        gap_score = {}
        for gap in [2**i for i in range(2, 15)]:
            valid_key_pairs = [key_pair for key_pair in [2, 4, 8, 16, 32, 64, 128, 256] if gap >= key_pair * 2]
            for key_pair in valid_key_pairs:
                gap = str(gap)
                num = str(key_pair)
                mark = f"D{num}_gap{gap}"
                print(mark)
                input_fpath = f"/public/home/ljt/tzc/data/MQAR/fixed_gaps/test_C8192_{mark}_in.pkl"
                if  "tiny" in model_config: 
                    model_name = model_config.split("-")[1]
                prediction_fpath = f"/public/home/ljt/tzc/evaluation/MQAR/{model_config}/{model_config}-based-lr1e3-{mark}/predictions.pkl"
                if not os.path.exists(prediction_fpath):   
                    prediction_fpath = f"/public/home/ljt/tzc/evaluation/MQAR/{model_config}/{model_name}_based_lr1e3-{mark}/version_3/predictions.pkl"
                    if not os.path.exists(prediction_fpath):
                        prediction_fpath = f"/public/home/ljt/tzc/evaluation/MQAR/{model_config}/{model_name}_based_lr1e3-{mark}/predictions.pkl"
                    else:
                        ...
                get_predictions(input_fpath, prediction_fpath)
        print(gap_score)
        auto_save_data(gap_score, root_save_dir + f"{model_config}_1e-3_score.pkl")
        draw_gap_score(gap_score, root_save_dir + f"{model_config}_1e-3_score.png" )