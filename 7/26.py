import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
import neurolab as nl  # 导入Neurolab库，用于神经网络构建和训练

# 定义一个函数，生成用于训练的数据
def get_data(num_points):
    # 生成四个不同的波形
    wave1 = 0.49 * np.sin(np.arange(0, num_points))
    wave2 = 3.62 * np.sin(np.arange(0, num_points))
    wave3 = 1.2 * np.sin(np.arange(0, num_points))
    wave4 = 4.6 * np.sin(np.arange(0, num_points))

    # 生成对应的振幅
    amp_1 = np.ones(num_points)
    amp_2 = 2 + np.zeros(num_points)
    amp_3 = 3.1 * np.zeros(num_points)
    amp_4 = 0.9 + np.zeros(num_points)

    # 将波形和振幅组合成输入数据
    wave = np.array([wave1, wave2, wave3, wave4]).reshape(num_points * 4, 1)
    amp = np.array([[amp_1, amp_2, amp_3, amp_4]]).reshape(num_points * 4, 1)

    return wave, amp


# 定义一个函数，用于可视化神经网络的输出
def visualize_output(nn, num_points_test):
    wave, amp = get_data(num_points_test)
    output = nn.sim(wave)
    plt.plot(amp.reshape(num_points_test * 4))
    plt.plot(output.reshape(num_points_test * 4))


if __name__ == '__main__':
    num_points = 50
    wave, amp = get_data(num_points)

    # 创建一个ELM（Extreme Learning Machine）神经网络模型
    nn = nl.net.newelm([[-3, 3]], [9, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
    nn.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    nn.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    nn.init()
    error_progress = nn.train(wave, amp, epochs=1200, show=100, goal=0.01)
    output = nn.sim(wave)

    # 绘制训练误差进展曲线
    plt.subplot(211)
    plt.plot(error_progress)
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')

    # 绘制原始数据和神经网络的预测结果
    plt.subplot(212)
    plt.plot(amp.reshape(num_points * 4))
    plt.plot(output.reshape(num_points * 4))
    plt.legend(['Original', 'Predicted'])

    # 创建新的图形窗口，并在窗口中绘制神经网络的输出
    plt.figure()
    plt.subplot(211)
    visualize_output(nn, 83)
    plt.xlim([0, 300])
    plt.subplot(212)
    visualize_output(nn, 48)
    plt.xlim([0, 300])
    plt.show()
