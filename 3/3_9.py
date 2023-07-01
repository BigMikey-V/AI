import numpy as np                         # 导入NumPy库，用于处理数组和数值计算
import matplotlib.pyplot as plt            # 导入Matplotlib库，用于绘图
from sklearn.cluster import KMeans         # 导入sklearn中的KMeans聚类算法

X = np.loadtxt('data_clustering.txt', delimiter=',')    # 从文件中加载数据到数组X

num_clusters = 5                          # 设置聚类的数量为5个

plt.figure()                              # 创建一个新的图形窗口
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
                                          # 绘制原始数据的散点图，圆形标记，无填充色，边框颜色为黑色，点的大小为80

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                                          # 计算图形的边界范围

plt.title('Input data')                   # 设置图形的标题为'Input data'
plt.xlim(x_min, x_max)                    # 设置x轴的范围
plt.ylim(y_min, y_max)                    # 设置y轴的范围
plt.xticks(())                            # 隐藏x轴刻度
plt.yticks(())                            # 隐藏y轴刻度

kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
                                          # 创建KMeans聚类对象，使用k-means++初始化方法，设置聚类数量为num_clusters，运行算法的次数为10次
kmeans.fit(X)                             # 对数据进行聚类

step_size = 0.01                          # 设置步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                                          # 计算图形的边界范围
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
                                          # 创建网格，用于绘制聚类边界
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
                                          # 对网格中的点进行聚类预测
output = output.reshape(x_vals.shape)      # 调整聚类输出的形状与网格一致

plt.figure()                              # 创建一个新的图形窗口
plt.clf()                                 # 清空当前图形
plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')
                                          # 绘制聚类边界的图像，使用颜色映射，自动调整纵横比例，原点位置为左下角
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
                                          # 绘制原始数据的散点图
cluster_centers = kmeans.cluster_centers_ # 获取聚类中心点的坐标
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', s=210, linewidths=4, color='black', zorder=12, facecolors='black')
                                          # 绘制聚类中心点的散点图

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                                          # 计算图形的边界范围
plt.title('Boundaries of clusters')       # 设置图形的标题为'Boundaries of clusters'
plt.xlim(x_min, x_max)                    # 设置x轴的范围
plt.ylim(y_min, y_max)                    # 设置y轴的范围
plt.xticks(())                            # 隐藏x轴刻度
plt.yticks(())                            # 隐藏y轴刻度

plt.show()                                # 显示图形
