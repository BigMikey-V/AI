import cv2  # 导入OpenCV库，用于图像处理和视频读取
import numpy as np  # 导入NumPy库，用于数值计算和数组操作

def get_frame(cap, scaling_factor):
    # 从视频捕获对象中获取当前帧
    _, frame = cap.read()

    # 根据缩放因子对帧进行缩放
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

if __name__ == '__main__':
    # 打开视频文件
    cap = cv2.VideoCapture('road.mp4')

    # 创建背景减除器对象
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # 设置背景减除器的历史帧数
    history = 100

    # 设置学习率，用于控制更新背景模型的速度
    learning_rate = 1.0 / history

    while True:
        # 获取当前帧
        frame = get_frame(cap, 0.25)

        # 应用背景减除器获取前景掩膜
        mask = bg_subtractor.apply(frame, learningRate=learning_rate)

        # 将掩膜转换为彩色图像
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 在原始帧和掩膜之间执行按位与操作，提取前景对象
        cv2.imshow('Input', frame)
        cv2.imshow('Output', mask & frame)

        # 等待按键事件，按下ESC键退出循环
        c = cv2.waitKey(10)
        if c == 27:
            break

    # 关闭窗口
    cv2.destroyAllWindows()
