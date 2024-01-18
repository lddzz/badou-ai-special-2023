import time

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img=cv2.imread('lenna.png')
h,w,c=img.shape[:3]  #有三个值

#灰度化
img_gray=np.zeros((h,w),img.dtype)
for i in range(h):
    for j in range(w):
            m=img[i,j]
            img_gray[i,j]=int(m[0]*0.33+m[1]*0.33+m[2]*0.33)  #BGR R-0.3 G-0.59 B-0.11

print(img_gray)
print("image show gray: %s"%img_gray)
# cv2.imshow("image show gray", img_gray) #用于在一个窗口中显示图像。
# cv2.waitKey(0)  #参数是等待时间（以毫秒为单位）。如果传递的参数是0，那么它将无限等待键盘输入。
# cv2.destroyAllWindows() #用于关闭所有由OpenCV创建的窗口。

plt.subplot(221)    #plt.subplot(221) 创建一个2x2的子图区域，并选择在这个区域中的第1个位置进行绘图
#img = plt.imread("lenna.png") #
plt.imshow(img_gray,cmap='gray') #读取的图像数据。这个函数会将数组中的数值映射到颜色空间，从而在图形窗口中呈现出图像的视觉表示

#灰度化,直接函数实现
img_gray=rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
print(img_gray)
print("image show gray: %s"%img_gray)

#二值化
img_binary = np.zeros((h, w), img.dtype)
for i in range(h):
    for j in range(w):
        if img_gray[i,j]<=0.5:
            img_binary[i,j]=0
        else:
            img_binary[i,j]=1 #1或者255都行

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
print(img_binary)
print("image show gray: %s"%img_binary)

plt.show()