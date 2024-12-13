from modelzipper.tutils import *
from rouge_score import rouge_scorer
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def main2():
    tmp = auto_read_data(f"/public/home/ljt/tzc/data/passkey_search/processed_data/128k_500_insert_ids.pkl")
    tokenizer = AutoTokenizer.from_pretrained("/public/home/ljt/hf_models/mamba-370m-hf")
    needle = 'eat a sandwich and sit in dolores park on a sunny day.'

    results = []
    for pred in tmp:
        pred_str = tokenizer.decode(pred['predictions'][0])[:64]
        depth, ctx_length = pred["depth"], pred["ctx_length"]
        # score = scorer.score(needle, pred_str)['rouge1'].fmeasure*10
        score = len(set(pred_str.split()).intersection(set(needle))) / len(needle)
        results.append({
            'ctx_length': ctx_length, 
            'depth': depth, 
            'score': score, 'pred': pred_str,
        })
        
    save_path = os.path.join("/nvme1/zecheng/evaluation/passkey_search/mamba-1_4b/passkey_search", f"ctx_up_32k_depth_0-1.jsonl")
    print_c(f"Saving at {save_path}", "yellow")
    auto_save_data(results, save_path)

    df = pd.DataFrame(results)
    df['depth'] = df['depth'].round(2)

    pivot_table = pd.pivot_table(df, values='score', index=['depth', 'ctx_length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="depth", columns="ctx_length", values="score") # This will turn into a proper pivot

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, 
        vmax=1,
        cmap=cmap,
        cbar=False,
        linewidths=0.5,
        linecolor='grey',
        linestyle='--'
    )

    title_font = {
        'fontsize': 16,
        'fontweight': 'bold',
        # 'fontname': 'Arial'
    }

    label_font = {
        'fontsize': 16,
        # 'fontweight': 'bold',
        # 'fontname': 'Arial'
    }

    x_values = df['ctx_length'].unique()
    x_ticks = x_values[5::6]  # take every 5 steps
    steps = list(range(6, len(x_values), 6))

    # 设置横坐标的位置和标签
    heatmap.set_xticks(steps)
    heatmap.set_xticklabels(x_ticks, rotation=0)

    ax = heatmap.get_figure().get_axes()[0]

    for j in steps:
        ax.axvline(x=j, color='black', linestyle=':', linewidth=1.5)

    ## More aesthetics
    plt.title('Passkey Search Results (Mamba 1.4B)', **title_font)  # Adds a title
    plt.xlabel('Context Length', **label_font)  # X-axis label
    plt.ylabel('Passkey Depth', **label_font)  # Y-axis label
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    save_path = os.path.join("/nvme1/zecheng/modelzipper/projects/state-space-model/analysis/passkey_search/raw_model", "mamba-1_4b-passkey_search_results.png")
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)
    

def main(ctx_lenth):

    tmp = auto_read_data(f"/nvme1/zecheng/evaluation/passkey_search/mamba-340m/patchify/pred_ctx_{ctx_lenth}_depth_0.pkl")

    tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/mamba-370m-hf")
    
    needle = 'eat a sandwich and sit in dolores park on a sunny day.'
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    results = []
    for pred in tmp:
        pred_str = pred['pred_res']
        replace_position, replace_depth = pred["replace_position"], pred["replace_depth"]
        # score = scorer.score(needle, pred_str)['rouge1'].fmeasure*10
        score = len(set(pred_str.split()).intersection(set(needle))) / len(needle)
        results.append({
            'replace_position': replace_position, 
            'replace_depth': replace_depth, 
            'score': score, 'pred': pred_str,
        })

    save_path = os.path.join("/nvme1/zecheng/evaluation/passkey_search/mamba-340m/patchify", f"eval_patchify_ctx_{ctx_lenth}_depth_0.jsonl")
    print_c(f"Saving at {save_path}", "yellow")
    auto_save_data(results, save_path)

    df = pd.DataFrame(results)

    pivot_table = pd.pivot_table(df, values='score', index=['replace_position', 'replace_depth'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="replace_position", columns="replace_depth", values="score") # This will turn into a proper pivot

    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, 
        vmax=1,
        cmap=cmap,
        cbar=False,
        linewidths=0.5,
        linecolor='grey',
        linestyle='--'
    )

    title_font = {
        'fontsize': 16,
        'fontweight': 'bold',
        # 'fontname': 'Arial'
    }

    label_font = {
        'fontsize': 16,
        # 'fontweight': 'bold',
        # 'fontname': 'Arial'
    }

    x_values = df['replace_depth'].unique()

    x_ticks = [f"{i * 20}%" for i in x_values]  # take every 5 steps
    steps = list(range(len(x_values))) 
    steps = [i + 0.5 for i in steps]
    # 设置横坐标的位置和标签
    heatmap.set_xticks(steps)
    heatmap.set_xticklabels(x_ticks, rotation=0)

    ax = heatmap.get_figure().get_axes()[0]

    for j in steps:
        ax.axvline(x=f"{j * 20}%", color='black', linestyle=':', linewidth=1.5)

    ## More aesthetics
    plt.title(f'Passkey Search Patchify {ctx_lenth}', **title_font)  # Adds a title
    plt.xlabel('Replaced Context Part', **label_font)  # X-axis label
    plt.ylabel('Replaced SSM State Index', **label_font)  # Y-axis label
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    save_path = os.path.join("/nvme1/zecheng/modelzipper/projects/state-space-model/analysis/passkey_search/patchify", f"passkey_patchify_{ctx_lenth}.png")
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)


if __name__ == "__main__":
    # CTX_LENGTH = [500, 1000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]
    # for ctx_len in CTX_LENGTH:
    #    main(ctx_len)
    main2()