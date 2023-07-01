import cv2  # 导入OpenCV库，用于图像处理和视频读取

def frame_diff(prev_frame, cur_frame, next_frame):
    # 计算当前帧和前一帧之间的差异
    diff_frames_1 = cv2.absdiff(next_frame, cur_frame)

    # 计算前一帧和当前帧之间的差异
    diff_frames_2 = cv2.absdiff(cur_frame, prev_frame)

    # 对两个差异帧进行逐像素与运算，获取两帧之间的差异区域
    return cv2.bitwise_and(diff_frames_1, diff_frames_2)

def get_frame(cap, scaling_factor):
    # 从视频捕获对象中获取当前帧
    _, frame = cap.read()

    # 根据缩放因子对帧进行缩放
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # 将帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray

if __name__ == '__main__':
    # 打开视频文件
    cap = cv2.VideoCapture('road.mp4')

    # 设置缩放因子
    scaling_factor = 0.4

    # 获取前一帧、当前帧和下一帧
    prev_frame = get_frame(cap, scaling_factor)
    cur_frame = get_frame(cap, scaling_factor)
    next_frame = get_frame(cap, scaling_factor)

    while True:
        # 计算当前帧和前一帧之间的差异，并显示差异图像
        cv2.imshow('Output', frame_diff(prev_frame, cur_frame, next_frame))

        # 更新前一帧、当前帧和下一帧
        prev_frame = cur_frame
        cur_frame = next_frame
        next_frame = get_frame(cap, scaling_factor)

        # 等待按键事件，按下ESC键退出循环
        key = cv2.waitKey(10)
        if key == 27:
            break

    # 关闭窗口
    cv2.destroyAllWindows()
