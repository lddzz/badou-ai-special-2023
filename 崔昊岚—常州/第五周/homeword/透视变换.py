import cv2
import numpy as np

img=cv2.imread('photo1.jpg')

result3=img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [340, 0], [0, 490], [340, 490]])

m=cv2.getPerspectiveTransform(src,dst)
result=cv2.warpPerspective(result3,m, (340,490))
cv2.imshow('src',img)
cv2.imshow('dst',result)
cv2.waitKey(0)