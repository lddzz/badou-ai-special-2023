import cv2
from numpy import shape
import random

def fun1(src,means,sigma,percentage):
    N_img=src.copy()
    h,w=src.shape[:2]
    N_num=int(percentage*h*w)
    for j in range(3):
        for i in range(N_num):
            x=random.randint(0,w-1)
            y=random.randint(0,h-1)
            N_img[x, y,j] = N_img[x, y,j] + random.gauss(means, sigma)
            if N_img[x, y,j]>255:
                N_img[x,y,j]=255
            elif N_img[x, y,j]<=0:
                N_img[x,y,j]=0
    return N_img

img=cv2.imread('lenna.png')
img1=fun1(img,2,4,0.8)
cv2.imshow('lenna',img)
cv2.imshow('gs',img1)
cv2.waitKey(0)

