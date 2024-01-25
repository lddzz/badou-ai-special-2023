import cv2
import matplotlib.pyplot as plt
from matplotlib import pyplot as pls

img=cv2.imread('lenna.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''




plt.figure()
plt.hist(gray.ravel(),256)

# gray.ravel是里面的每一个元素
# for i in gray.ravel():
#     print(i)


'''
plt.hist() 函数用于绘制直方图，它的基本用法如下：

python
Copy code
import matplotlib.pyplot as plt

# data 是待绘制直方图的数据，可以是一维数组
# bins 指定直方图的条形数
plt.hist(data, bins=10)

# 显示直方图
plt.show()
在上面的代码中：

data: 是待绘制直方图的数据，可以是一维数组。在你的代码中，dst.ravel() 返回的一维数组就是用作直方图的数据。

bins: 指定直方图的条形数，即柱子的数量。在你的代码中，bins=256 表示将灰度级别分成 256 个区间，每个区间对应直方图中的一个柱子。

'''


plt.show()