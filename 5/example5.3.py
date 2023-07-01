import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
from scipy.io.wavfile import write  # 导入write函数，用于写入音频文件

output_file = 'generated_audio.wav'  # 设置输出音频文件的文件名

duration = 4  # 设置音频时长（秒）
sampling_freq = 44100  # 设置采样频率（每秒采样点数）
tone_freq = 784  # 设置音调频率（Hz）
min_val = -4 * np.pi  # 设置时间轴的起始值
max_val = 4 * np.pi  # 设置时间轴的结束值

t = np.linspace(min_val, max_val, duration * sampling_freq)  # 生成时间轴上的值
signal = np.sin(2 * np.pi * tone_freq * t)  # 生成正弦波信号

noise = 0.5 * np.random.rand(duration * sampling_freq)  # 生成噪声信号
signal += noise  # 将噪声信号添加到正弦波信号中

scaling_factor = np.power(2, 15) - 1  # 设置缩放因子，用于将信号归一化到[-1, 1]范围内
signal_normalized = signal / np.max(np.abs(signal))  # 归一化信号
signal_scaled = np.int16(signal_normalized * scaling_factor)  # 缩放信号并转换为16位整数形式

write(output_file, sampling_freq, signal_scaled)  # 将信号写入音频文件

signal = signal[:200]  # 只取前200个采样点用于绘制波形图

time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq)  # 计算时间轴上的值（毫秒）

plt.plot(time_axis, signal, color='black')  # 绘制波形图
plt.xlabel('Time (milliseconds)')  # 设置横坐标标签
plt.ylabel('Amplitude')  # 设置纵坐标标签
plt.title('Generated audio signal')  # 设置图形标题
plt.show()  # 显示图形
