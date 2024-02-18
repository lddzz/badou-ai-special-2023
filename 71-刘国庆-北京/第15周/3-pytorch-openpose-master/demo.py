# 导入所需的库
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src import util
from src.body import Body
from src.hand import Hand

# 加载人体和手部姿势模型
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')
# 使用OpenCV读取图像，注意OpenCV读取的图像颜色顺序是BGR
image = cv2.imread('images/star.png')
# 使用人体姿势模型对图像进行处理，得到候选关键点candidate和子集subset
candidate, subset = body_estimation(image)
# 复制原始图像，用于绘制姿势
canvas = copy.deepcopy(image)
# 在复制的图像上绘制人体姿势
# 参数说明：
# canvas：用于绘制姿势的图像
# candidate： 候选关键点
# subset：子集
canvas = util.draw_bodypose(canvas, candidate, subset)
# 使用手部检测函数检测手部位置hands_list
# candidate：候选关键点
# subset：子集
# image：原始图像
hand_list = util.handDetect(candidate, subset, image)
# 初始化一个空列表，用于存储所有手部的关键点
all_hand_peaks = []
# 遍历检测到的每一个手部
# x：手部区域的左上角x坐标，
# y：手部区域的左上角y坐标，
# w：手部区域的宽度，
# is_left：是否是左手，True表示是左手，False表示是右手
# hands_list：检测到的手部列表
for x, y, w, is_left in hand_list:
    # 使用手部姿势模型对手部区域进行处理，得到手部关键点peaks
    # image[y:y + w, x:x + w, :]：手部区域
    peaks = hand_estimation(image[y:y + w, x:x + w, :])
    # 将关键点的坐标转换为相对于原始图像的坐标
    # 将关键点的x坐标从相对于手部区域的坐标转换为相对于原始图像的坐标
    # 如果关键点的x坐标等于0（可能表示该关键点未被检测到），则保持不变；否则，将x坐标加上手部区域的左上角x坐标
    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
    # 将关键点的y坐标从相对于手部区域的坐标转换为相对于原始图像的坐标
    # 如果关键点的y坐标等于0（可能表示该关键点未被检测到），则保持不变；否则，将y坐标加上手部区域的左上角y坐标
    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
    # 将手部关键点添加到列表中
    all_hand_peaks.append(peaks)
# 在复制的图像上绘制手部姿势
canvas = util.draw_handpose(canvas, all_hand_peaks)
# 使用matplotlib显示处理后的图像，注意颜色顺序需要转换为RGB
# canvas[:, :, [2, 1, 0]]：将BGR颜色顺序转换为RGB颜色顺序
plt.imshow(canvas[:, :, [2, 1, 0]])
# 隐藏坐标轴
plt.axis('off')
# 显示图像
plt.show()
