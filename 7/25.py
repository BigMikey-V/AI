import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
import neurolab as nl  # 导入Neurolab库，用于神经网络构建和训练

# 定义数据的范围和采样点数量
min_val = -20
max_val = 20
num_points = 150

# 在指定范围内生成num_points个等间隔采样点，作为x的值
x = np.linspace(min_val, max_val, num_points)
# 根据函数y = 2x^2 + 7生成对应的y值
y = 2 * np.square(x) + 7
# 对y进行归一化处理，使其范围在0到1之间
y /= np.linalg.norm(y)

# 将x和y转换为二维数组形式，用于训练神经网络
data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

# 绘制输入数据散点图
plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# 创建神经网络模型，输入层有1个神经元，隐藏层有10个和6个神经元，输出层有1个神经元
nn = nl.net.newff([[min_val, max_val]], [10, 6, 1])
# 设置使用梯度下降法（Gradient Descent）进行训练
nn.trainf = nl.train.train_gd
# 训练神经网络，并获取训练误差的进展
error_progress = nn.train(data, labels, epochs=1200, show=100, goal=0.01)

# 绘制训练误差进展曲线
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')

# 在更密集的x值范围内进行预测，并将预测结果绘制出来
x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)

# 绘制实际曲线、神经网络预测曲线和原始数据点
plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs predicted')
plt.show()
