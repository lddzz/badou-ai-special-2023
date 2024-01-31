import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像 'lenna.png' 使用 OpenCV
img = cv2.imread('lenna.png')

# 将图像转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图应用直方图均衡化
dst = cv2.equalizeHist(gray)

# 计算均衡化图像的直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

# 使用 matplotlib 绘制直方图
plt.figure()
plt.hist(dst.ravel(), bins=256)
plt.show()

# 显示原始灰度图和均衡化图像并排
cv2.imshow('His EQ', np.hstack((gray, dst)))

# 等待按键事件以关闭 OpenCV 窗口
cv2.waitKey(0)
