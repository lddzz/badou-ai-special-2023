import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('lenna.png',0)
print(img.shape)

rows,cols=img.shape[:]

data=img.reshape((rows*cols,1))
data=np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10,1.0)

flags=cv2.KMEANS_RANDOM_CENTERS

compactness,labels,centers=cv2.kmeans(data,4,None,criteria,10,flags)


#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(dst,cmap='gray')



plt.show()