import torch
import numpy as np
import matplotlib.pyplot as plt

# 假设我们有一个定义好的一维卷积核，这里随机生成一个
# in_channels = 1, out_channels = 1, kernel_size = 50
kernel_size = 50
conv_filter = torch.randn(1, 1, kernel_size)

# 快速傅里叶变换(FFT)来分析滤波器中不同频率的成分
filter_fft = torch.fft.fft(conv_filter)

import pdb; pdb.set_trace()

# 对FFT结果取绝对值得到大小谱(magnitude spectrum)，且通常会取对数的形式显示
magnitude_spectrum = torch.log(torch.abs(filter_fft) + 1e-8)  # 防止对数为负无穷

# 因为FFT的结果是对称的，只需取一半的频率即可
half_length = kernel_size // 2 + 1
frequencies = torch.arange(half_length)

# 从tensor转为numpy用于绘图
magnitude_spectrum_np = magnitude_spectrum[0, 0, :half_length].detach().numpy()


# 绘制频谱图
y_values_np = magnitude_spectrum_np.real

# 绘制频谱图
plt.figure(figsize=(10, 5))
plt.plot(frequencies.cpu().numpy(), y_values_np)  # 修改为plot以避免复杂的stem处理
plt.title('Magnitude Spectrum of the Convolution Filter')
plt.xlabel('Frequency')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.savefig("/nvme1/zecheng/modelzipper/projects/state-space-model/back/plot.png")
