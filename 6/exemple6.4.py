import cv2  # 导入OpenCV库，用于图像处理和视频读取
import numpy as np  # 导入NumPy库，用于数值计算和数组操作

# 打开视频文件
cap = cv2.VideoCapture('face.mp4')

# 从视频捕获对象中读取一帧
_, frame = cap.read()

# 将帧从BGR颜色空间转换为HSV颜色空间
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 创建掩膜，通过指定颜色范围来过滤出感兴趣的区域
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

# 设置追踪窗口的初始位置
x0, y0, x1, y1 = 200, 100, 300, 400
track_window = (x0, y0, x1, y1)

# 提取感兴趣区域（ROI）
roi = frame[y0:y0+y1, x0:x0+x1]

# 计算ROI的直方图
hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# 归一化直方图
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

# 设置追踪终止的条件
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    # 从视频捕获对象中读取一帧
    _, frame = cap.read()

    # 缩放帧的大小
    scaling_factor = 0.5
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # 将帧从BGR颜色空间转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 使用反向投影算法得到目标位置的概率分布图
    dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)

    # 使用CamShift算法进行目标跟踪
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # 获取跟踪窗口的四个顶点坐标
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)

    # 在原始帧上绘制跟踪结果
    img = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Output', img)

    # 等待按键事件，按下ESC键退出循环
    key = cv2.waitKey(5)
    if key == 27:
        break

# 释放视频捕获对象
cap.release()

# 关闭窗口
cv2.destroyAllWindows()
