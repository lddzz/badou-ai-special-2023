import cv2
import numpy as np

def fun(img):
    h,w,c=img.shape
    nimg=np.zeros((800,800,3),dtype=np.uint8)
    rate=h/800
    print(rate)
    for i in range(800):
        for j in range(800):
            nimg[i,j]=img[int(i*rate+0.5),int(j*rate+0.5)]
    return nimg

img = cv2.imread('lenna.png')
zoom=fun(img)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)

