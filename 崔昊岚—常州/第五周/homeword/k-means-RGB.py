import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('lenna.png')

data=img.reshape((-1,3))
data=np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10,1.0)

flags=cv2.KMEANS_RANDOM_CENTERS


for i in range(1,7):
    compactness,labels,centers=cv2.kmeans(data,2**i,None,criteria,10,flags)
    centers=np.uint8(centers)  #因为centers可能不是整数了
    res=centers[labels.flatten()]
    '''
    也就是说选择的中心的值
    '''

    dst=res.reshape((img.shape))
    dst=cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i), plt.imshow(dst, 'gray')
plt.show()