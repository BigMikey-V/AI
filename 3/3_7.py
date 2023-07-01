# 导入pandas库并简写为pd，用于数据处理
import pandas as pd
# 导入numpy库并简写为np，用于处理数值数据
import numpy as np
# 导入matplotlib库中的pyplot模块并简写为plt，用于绘图
import matplotlib.pyplot as plt
# 从sklearn库中导入datasets模块，用于加载数据集
from sklearn import datasets
# 从sklearn库中导入LinearRegression类，用于线性回归算法

# 使用datasets模块中的load_boston函数加载波士顿房价数据集
dataset = datasets.load_boston()
# 获取数据集的特征数据，存储在变量x中
x = dataset.data
# 获取数据集的目标值（房价），存储在变量y中
y = dataset.target
# 获取特征的名称，存储在变量names中
names = dataset.feature_names

# 循环遍历13个特征
for i in range(13):
    # 绘制子图，subplot函数的参数表示子图的行数、列数和当前子图的索引
    plt.plot(7, 2, i + 1)
    # 绘制散点图，x轴为第i个特征的取值，y轴为房价y的取值
    plt.scatter(x[:, i], y, s=10)
    # 设置子图的标题为特征的名称
    plt.title(names[i])

# 显示所有子图
plt.show()
