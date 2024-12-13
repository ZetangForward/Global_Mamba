import torch

def sample_positions(N, k, x):
    # 确保参数是合理的
    assert k * x <= N, "Not enough space to fit k positions with minimum gap x in sequence of length N"

    # 创建一个长度为 N - (k - 1) * x 的序列
    max_range = N - (k - 1) * x

    # 从这个序列中随机选择 k 个位置
    random_positions = torch.randperm(max_range)[:k]

    # 为每个位置添加偏移量
    offsets = torch.arange(k) * x
    sampled_positions = random_positions + offsets

    # 对最终的位置进行排序
    sampled_positions, _ = torch.sort(sampled_positions)

    return sampled_positions

# 使用示例
N = 100  # 序列的总长度
k = 5    # 要采样的位置数
x = 10   # 位置之间的最小间隔
sampled_positions = sample_positions(N, k, x)

print(sampled_positions)
