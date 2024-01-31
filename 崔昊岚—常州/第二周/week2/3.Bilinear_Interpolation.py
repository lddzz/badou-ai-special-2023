import cv2
import numpy as np

def fun(img):
    f=0
    h,w,c=img.shape
    nimg=np.zeros((800,800,3),dtype=np.uint8)
    rate=float(h)/800
    print(rate)
    for i in range(800):
        for j in range(800):

            s_x=(i+0.5)*rate-0.5
            s_y = (j + 0.5) * rate - 0.5

            s_x1=int(s_x)
            s_x2=min(s_x1 + 1 ,h - 1)
            s_y1=int(s_y)
            s_y2 = min(s_y1 + 1, w - 1)

            s_1 = img[s_x1,s_y1]
            s_2 = img[s_x2,s_y1]
            s_3 = img[s_x1,s_y2]
            s_4 = img[s_x2,s_y2]

            rate_x1=s_1*(s_x2-s_x)+s_2*(s_x-s_x1)
            rate_x2=s_3*(s_x2-s_x)+s_4*(s_x-s_x1)

            rate_xy=rate_x1*(s_y2-s_y)+rate_x2*(s_y-s_y1)
            nimg[i,j]=rate_xy
            f = f + 1
            print(f)


    return nimg

img = cv2.imread('lenna.png')
zoom=fun(img)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)

