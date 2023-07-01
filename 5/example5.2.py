import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
from scipy.io import wavfile  # 导入wavfile模块，用于读取音频文件

sampling_freq, signal = wavfile.read('spoken_word.wav')  # 读取音频文件，获取采样频率和信号数据

signal = signal / np.power(2, 15)  # 对信号进行归一化，将其范围缩放到[-1, 1]

len_signal = len(signal)  # 获取信号的长度

len_half = np.ceil((len_signal + 1) / 2.0).astype(np.int)  # 计算频谱的一半长度

freq_signal = np.fft.fft(signal)  # 对信号进行傅里叶变换，得到频域信号

freq_signal = abs(freq_signal[0:len_half]) / len_signal  # 计算频域信号的幅度谱

freq_signal **= 2  # 对幅度谱进行平方

len_fts = len(freq_signal)  # 获取幅度谱的长度

if len_signal % 2:
    freq_signal[1:len_fts] *= 2
else:
    freq_signal[1:len_fts-1] *= 2
# 根据信号的长度是否为奇数，对频域信号进行调整，以修正功率谱的计算

signal_power = 10 * np.log10(freq_signal)  # 计算信号的功率谱（以对数形式表示）

x_axis = np.arange(0, len_half, 1) * (sampling_freq / len_signal) / 1000.0  # 计算频率轴上的值

plt.figure()  # 创建一个新的图形窗口

plt.plot(x_axis, signal_power, color='black')  # 绘制频谱图
plt.xlabel('Frequency (kHz)')  # 设置横坐标标签
plt.ylabel('Signal power(dB)')  # 设置纵坐标标签
plt.show()  # 显示图形
