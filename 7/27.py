import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import neurolab as nl  # 导入Neurolab库，用于神经网络构建和训练

input_file = 'letter.data'  # 输入数据文件名
num_datapoints = 40  # 数据点数目
orig_labels = 'encode'  # 原始标签
num_orig_labels = len(orig_labels)  # 原始标签的数量

num_train = int(0.9 * num_datapoints)  # 训练数据数目
num_test = num_datapoints - num_train  # 测试数据数目
start = 6  # 数据起始索引
end = -1  # 数据结束索引
data = []  # 存储特征数据的列表
labels = []  # 存储标签数据的列表

with open(input_file, 'r') as f:  # 打开输入数据文件
    for line in f.readlines():  # 逐行读取数据
        list_val = line.split('\t')  # 按制表符分割每行数据
        if list_val[1] not in orig_labels:  # 如果标签不在原始标签列表中，则跳过该行数据
            continue
        label = np.zeros((num_orig_labels, 1))  # 创建标签数组
        label[orig_labels.index(list_val[1])] = 1  # 将标签对应位置置为1，进行独热编码
        labels.append(label)  # 将标签添加到列表中
        cur_char = np.array([float(x) for x in list_val[start:end]])  # 将特征数据转换为浮点数，并存储为数组
        data.append(cur_char)  # 将特征数据添加到列表中
        if len(data) > num_datapoints:  # 如果数据点数目达到指定数量，则停止读取
            break
    data = np.asfarray(data)  # 将特征数据转换为浮点类型的NumPy数组
    labels = np.array(labels).reshape(num_datapoints, num_orig_labels)  # 将标签数据转换为NumPy数组，并进行形状调整
    num_dims = len(data[0])  # 特征维度数目

    nn = nl.net.newff([[0, 1] for _ in range(len(data[0]))], [128, 16, num_orig_labels])  # 创建一个前馈神经网络模型
    nn.trainf = nn.train.train_gd  # 设置网络的训练函数为梯度下降
    error_progress = nn.train(data[:num_train, :], labels[:num_train, :], epochs=5000, show=500, goal=0.01)  # 使用训练数据进行训练
    print('\nTesting on unknown data:')
    predicted_test = nn.sim(data[num_train:, :])  # 使用测试数据进行预测
    for i in range(num_test):
        print('\nOriginal:', orig_labels[np.argmax(labels[i])])  # 打印原始标签
        print('\nPredicted:', orig_labels[np.argmax(predicted_test[i])])  # 打印预测标签
