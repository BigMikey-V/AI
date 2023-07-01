# 从sklearn库中导入load_iris函数，用于加载鸢尾花数据集
from sklearn.datasets import load_iris
# 导入pandas库并简写为pd，用于数据处理
import pandas as pd
# 从sklearn库中导入LogisticRegression类，用于逻辑回归算法
from sklearn.linear_model import LogisticRegression
# 导入numpy库并简写为np，用于处理数值数据
import numpy as np
# 导入matplotlib库并简写为plt，用于绘图
import matplotlib.pyplot as plt
# 从sklearn库中导入train_test_split函数，用于将数据集分为训练集和测试集
from sklearn.model_selection import train_test_split

# 使用load_iris函数加载鸢尾花数据集，iris是一个类似字典的对象，包含特征数据和标签
iris = load_iris()
# 获取鸢尾花数据集的特征数据，存储在变量x中
x = iris.data
# 获取鸢尾花数据集的标签，存储在变量y中
y = iris.target

# 使用train_test_split函数将数据集分为训练集和测试集
# random_state参数用于设定随机种子，保证每次运行得到的结果一致
# test_size参数用于设定测试集的比例，这里设置为0.20表示测试集占总数据集的20%
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.20)

# 创建LogisticRegression实例clf，C参数表示正则化强度，solver参数指定优化算法，multi_class参数指定多类分类方法
clf = LogisticRegression(C=1, solver='newton-cg', multi_class='multinomial')

# 使用fit函数对模型进行训练，传入训练集的特征数据x_train和对应的标签y_train
clf.fit(x_train, y_train)

# 输出测试集的真实标签y_test
print("实际值:", y_test)
# 使用训练好的模型对测试集进行预测，并输出预测结果
print("预测值:", clf.predict(x_test))

# 输出训练集上的模型准确率
print(clf.score(x_train, y_train))
# 输出测试集上的模型准确率
print(clf.score(x_test, y_test))

# 使用训练好的模型对新的样本进行预测
print(clf.predict([[3.1, 2.3, 1.2, 0.5]]))
