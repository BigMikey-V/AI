import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
from scipy.io import wavfile  # 导入wavfile模块，用于读取音频文件
from python_speech_features import mfcc, logfbank  # 导入python_speech_features库，用于提取语音特征

sampling_freq, signal = wavfile.read('random_sound.wav')  # 从音频文件中读取采样频率和信号数据

signal = signal[:10000]  # 截取前10000个采样点用于特征提取

features_mfcc = mfcc(signal, sampling_freq)  # 提取MFCC特征

print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])  # 输出MFCC特征的窗口数
print('Length of each feature =', features_mfcc.shape[1])  # 输出每个MFCC特征的长度

features_mfcc = features_mfcc.T  # 转置MFCC特征矩阵，使每列代表一个特征维度
plt.matshow(features_mfcc)  # 绘制MFCC特征矩阵的热图
plt.title('MFCC')  # 设置图形标题

features_fb = logfbank(signal, sampling_freq)  # 提取滤波器组特征

print('\nFilter bank:\nNumber of windows =', features_fb.shape[0])  # 输出滤波器组特征的窗口数
print('Length of each feature =', features_fb.shape[1])  # 输出每个滤波器组特征的长度

features_fb = features_fb.T  # 转置滤波器组特征矩阵，使每列代表一个特征维度
plt.matshow(features_fb)  # 绘制滤波器组特征矩阵的热图
plt.title('Filter bank')  # 设置图形标题
plt.show()  # 显示图形
