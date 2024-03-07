import cv2
import numpy as np

def fun(img1_gray,kp1,img2_gray,kp2,goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


img1=cv2.imread('c1.jpg')
img2=cv2.imread('c2.jpg')

#初始化sift
sift=cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#找到2中距离1中每个点最近的两个点
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1,des2,k=2)

good_matches = []

for m, n in matches:
    if m.distance <0.5*n.distance:
        good_matches.append(m)

fun(img1,kp1,img2,kp2,good_matches[:30])

cv2.waitKey(0)
cv2.destroyAllWindows()