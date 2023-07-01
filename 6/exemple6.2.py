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

    # 设置缩放因子
    scaling_factor = 0.25

    while True:
        # 获取当前帧
        frame = get_frame(cap, scaling_factor)

        # 将帧从BGR颜色空间转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 设置颜色范围，提取指定颜色范围内的像素
        lower = np.array([0, 40, 40])
        upper = np.array([150, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # 对原始帧和掩膜进行按位与操作，保留指定颜色范围内的像素
        img_bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)

        # 对图像进行中值模糊处理，以减少噪声
        img_median_blurred = cv2.medianBlur(img_bitwise_and, 5)

        # 显示原始帧和处理后的帧
        cv2.imshow('Input', frame)
        cv2.imshow('Output', img_median_blurred)

        # 等待按键事件，按下ESC键退出循环
        c = cv2.waitKey(5)
        if c == 27:
            break

    # 关闭窗口
    cv2.destroyAllWindows()
