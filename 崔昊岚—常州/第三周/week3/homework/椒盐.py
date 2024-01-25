import cv2
from numpy import shape
import random

def fun1(src,percentage):
    N_img=src
    h,w=src.shape[:2]
    N_num=int(percentage*h*w)
    for i in range(N_num):
        x=random.randint(0,w-1)
        y=random.randint(0,h-1)
        if random.random()<=0.5:
            N_img[x,y]=255
        else:
            N_img[x,y]=0
    return N_img

img=cv2.imread('lenna.png')
img1=fun1(img,0.2)

cv2.imshow('lenna',img)
cv2.imshow('p_s',img1)
cv2.waitKey(0)

