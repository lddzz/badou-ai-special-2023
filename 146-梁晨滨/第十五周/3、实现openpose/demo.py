import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np


from src import util
from src.body import Body
from src.hand import Hand

# 身体和手的预测网络
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

test_image = 'images/person.jpg'
oriImg = cv2.imread(test_image)  # B,G,R order
canvas = copy.deepcopy(oriImg)
# 检测身体并画框
candidate, subset = body_estimation(oriImg)
canvas = util.draw_bodypose(canvas, candidate, subset)

# 检测手(根据上面身体中的部分点大致定位手)
all_hand_peaks = []
hands_list = util.handDetect(candidate, subset, oriImg)

# 根据大致定位的手进行手部检测
for x, y, w, is_left in hands_list:

    peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)

    all_hand_peaks.append(peaks)

canvas = util.draw_handpose(canvas, all_hand_peaks)

plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
