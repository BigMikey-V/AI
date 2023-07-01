import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
import neurolab as nl  # 导入Neurolab库，用于神经网络构建和训练

# 创建输入数据
data = np.array([[0.2, 0.3], [0.5, 0.4], [0.4, 0.6], [0.7, 0.5]])
# 创建标签数据
labels = [[0], [0], [0], [1]]

# 绘制输入数据散点图
plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# 定义输入维度范围
dim1 = [0, 1]
dim2 = [0, 1]
# 定义输出维度数量
num_output = 1
# 创建感知器（perceptron）模型
perceptron = nl.net.newp([dim1, dim2], num_output)
# 训练感知器模型并获取训练误差的进展
error_progress = perceptron.train(data, labels, epochs=80, show=20, lr=0.03)

# 绘制训练误差进展曲线
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Train error')
plt.title('Training error progress')
plt.grid()
plt.show()
