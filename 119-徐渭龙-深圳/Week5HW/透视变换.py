import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

resultImg= img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])  #顶点坐标
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]]) #目标图像的顶点坐标
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)  # 得到transformer透视函数
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(resultImg, m, (337, 488)) #映射到结果目标图像
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
