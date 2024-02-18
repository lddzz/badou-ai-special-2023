import math  # 导入Python的math库，提供数学函数和常量
import cv2  # 导入OpenCV库，这是一个用于处理图像和视频的库
import matplotlib  # 导入matplotlib库，这是一个用于创建静态、动态、交互式图表的库
import matplotlib.pyplot as plt  # 从matplotlib库中导入pyplot模块，这是一个用于创建图表和绘图的模块
import numpy as np  # 导入numpy库并将其别名为np，这是一个用于处理大型多维数组和矩阵的库，提供大量的数学函数来操作这些数组
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # 从matplotlib的后端导入FigureCanvasAgg，并将其别名为FigureCanvas，这是一个用于在Agg画布上绘制图形的类
from matplotlib.figure import Figure  # 从matplotlib的figure模块导入Figure类，这是一个用于创建新图形的类


# 定义函数，接受三个参数：图像、步长和填充值
def padRightDownCorner(img, stride, padValue):
    # 获取图像的高度
    h = img.shape[0]
    # 获取图像的宽度
    w = img.shape[1]
    # 初始化一个长度为4的列表，用于存储四个方向的填充值
    pad = 4 * [None]
    # 上方不填充
    pad[0] = 0
    # 左方不填充
    pad[1] = 0
    # 计算下方需要填充的值，如果高度可以被步长整除，则不填充，否则填充的值为步长减去高度除以步长的余数
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    # 计算右方需要填充的值，如果宽度可以被步长整除，则不填充，否则填充的值为步长减去宽度除以步长的余数
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right
    # 初始化填充后的图像为原图像
    img_padded = img
    # 创建上方的填充区域
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    # 将上方的填充区域添加到图像上方
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    # 创建左方的填充区域
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    # 将左方的填充区域添加到图像左方
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    # 创建下方的填充区域
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    # 将下方的填充区域添加到图像下方
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    # 创建右方的填充区域
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    # 将右方的填充区域添加到图像右方
    img_padded = np.concatenate((img_padded, pad_right), axis=1)
    # 返回填充后的图像和填充值列表
    return img_padded, pad


# 定义函数，接受两个参数：当前模型和预训练的模型权重
def transfer(model, model_weights):
    # 初始化一个空字典，用于存储转移后的模型权重
    transfered_model_weights = {}
    # 遍历当前模型的所有权重名称
    for weights_name in model.state_dict().keys():
        # 将预训练模型的权重转移到当前模型中，注意这里假设预训练模型的权重名称和当前模型的权重名称有一定的对应关系
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    # 返回转移后的模型权重
    return transfered_model_weights


# 定义函数，接受三个参数：画布、候选关键点和子集
def draw_bodypose(canvas, candidate, subset):
    # 设置连线的宽度
    stickwidth = 4
    # 定义人体各关键点之间的连线顺序
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11],
        [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18],
        [3, 17], [6, 18]
    ]
    # 定义各关键点的颜色
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]
    # 遍历所有关键点，绘制关键点
    # 遍历18个关键点
    for i in range(18):
        # 遍历每个人的关键点集合
        for n in range(len(subset)):
            # 获取当前关键点的索引
            index = int(subset[n][i])
            # 如果索引为-1，表示该关键点不存在，跳过当前循环
            if index == -1:
                continue
            # 获取关键点的x和y坐标
            x, y = candidate[index][0:2]
            # 在画布上绘制关键点，颜色为预定义的颜色，大小为4，填充
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    # 遍历所有关键点，绘制连线
    # 遍历17条肢体连线
    for i in range(17):
        # 遍历每个人的关键点集合
        for n in range(len(subset)):
            # 获取当前连线的两个关键点的索引
            index = subset[n][np.array(limbSeq[i]) - 1]
            # 如果索引中存在-1，表示该连线的某个关键点不存在，跳过当前循环
            if -1 in index:
                continue
            # 复制当前画布
            cur_canvas = canvas.copy()
            # 获取两个关键点的y坐标
            Y = candidate[index.astype(int), 0]
            # 获取两个关键点的x坐标
            X = candidate[index.astype(int), 1]
            # 计算两个关键点的x坐标的平均值
            mX = np.mean(X)
            # 计算两个关键点的y坐标的平均值
            mY = np.mean(Y)
            # 计算两个关键点之间的距离
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            # 计算两个关键点连线的角度
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            # 根据关键点的坐标、连线的长度和角度，生成一个多边形
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            # 在复制的画布上填充生成的多边形，颜色为预定义的颜色
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            # 将原画布和填充了多边形的画布进行加权合并，原画布的权重为0.4，填充了多边形的画布的权重为0.6，gamma值为0
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # 返回绘制完成的画布
    return canvas


# 定义函数，接受三个参数：画布、所有手部关键点和是否显示编号的标志
def draw_handpose(canvas, all_hand_peaks, show_number=False):
    # 定义手部各关键点之间的连线顺序
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    # 创建一个新的图形，大小与画布相同
    fig = Figure(figsize=plt.figaspect(canvas))
    # 调整子图的位置，使其填充整个图形
    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    # 创建一个新的画布
    bg = FigureCanvas(fig)
    # 创建一个新的子图
    ax = fig.subplots()
    # 关闭坐标轴
    ax.axis('off')
    # 在子图上显示画布
    ax.imshow(canvas)
    # 获取图形的宽度和高度
    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()
    # 遍历所有手部关键点
    for peaks in all_hand_peaks:
        # 遍历所有连线
        for ie, e in enumerate(edges):
            # 如果连线的两个关键点都存在
            if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
                # 获取连线的两个关键点的坐标
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                # 在子图上绘制连线，颜色为HSV颜色空间的颜色
                ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]))
        # 遍历所有关键点
        for i, keyponit in enumerate(peaks):
            # 获取关键点的坐标
            x, y = keyponit
            # 在子图上绘制关键点，颜色为红色
            ax.plot(x, y, 'r.')
            # 如果设置了显示编号的标志，则在关键点旁边显示编号
            if show_number:
                ax.text(x, y, str(i))
    # 绘制图形
    bg.draw()
    # 将画布转换为RGB图像
    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    # 返回绘制完成的画布
    return canvas


# 定义函数，接受三个参数：画布、所有手部关键点和是否显示编号的标志
def draw_handpose_by_opencv(canvas, peaks, show_number=False):
    # 定义手部各关键点之间的连线顺序
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    # 遍历所有连线
    for ie, e in enumerate(edges):
        # 如果连线的两个关键点都存在
        if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
            # 获取连线的两个关键点的坐标
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            # 在画布上绘制连线，颜色为HSV颜色空间的颜色，线宽为2
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255,
                     thickness=2)
    # 遍历所有关键点
    for i, keyponit in enumerate(peaks):
        # 获取关键点的坐标
        x, y = keyponit
        # 在画布上绘制关键点，颜色为红色，大小为4，填充
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        # 如果设置了显示编号的标志，则在关键点旁边显示编号
        if show_number:
            cv2.putText(canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), lineType=cv2.LINE_AA)
    # 返回绘制完成的画布
    return canvas


# 定义函数检测手部的位置，接受三个参数：候选关键点、子集和原始图像
def handDetect(candidate, subset, oriImg):
    # 设置手腕和肘部的比例为0.33
    ratioWristElbow = 0.33
    # 初始化一个空列表，用于存储检测结果
    detect_result = []
    # 获取原始图像的高度和宽度
    image_height, image_width = oriImg.shape[0:2]
    # 遍历子集中的每个人
    for person in subset.astype(int):
        # 判断该人是否有左手，如果左肩、左肘和左手腕的索引都不为-1，则表示有左手
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        # 判断该人是否有右手，如果右肩、右肘和右手腕的索引都不为-1，则表示有右手
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        # 如果该人既没有左手也没有右手，则跳过当前循环
        if not (has_left or has_right):
            continue
        # 初始化一个空列表，用于存储手部的信息
        hands = []
        # 如果有左手，则获取左肩、左肘和左手腕的坐标，并将这些坐标和一个表示左手的标志添加到手部信息列表中
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # 如果有右手，则获取右肩、右肘和右手腕的坐标，并将这些坐标和一个表示右手的标志添加到手部信息列表中
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        # 遍历手部信息列表中的每个手部
        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # 计算手腕的x坐标和y坐标，这里假设手腕和肘部的距离是肘部和手腕的距离的0.33倍
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            # 计算手腕和肘部的距离
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            # 计算肘部和肩部的距离
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # 计算宽度，这里假设宽度是手腕和肘部的距离的1.5倍和肘部和肩部的距离的0.9倍中的最大值
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # 将x坐标和y坐标向左和向上移动宽度的一半，以使得手部位于检测框的中心
            x -= width / 2
            y -= width / 2
            # 如果x坐标小于0，则将x坐标设置为0
            if x < 0: x = 0
            # 如果y坐标小于0，则将y坐标设置为0
            if y < 0: y = 0
            # 初始化宽度1和宽度2为宽度
            width1 = width
            width2 = width
            # 如果x坐标加上宽度大于图像的宽度，则将宽度1设置为图像的宽度减去x坐标
            if x + width > image_width: width1 = image_width - x
            # 如果y坐标加上宽度大于图像的高度，则将宽度2设置为图像的高度减去y坐标
            if y + width > image_height: width2 = image_height - y
            # 将宽度设置为宽度1和宽度2中的最小值
            width = min(width1, width2)
            # 如果宽度大于等于20，则将x坐标、y坐标、宽度和是否为左手的标志添加到检测结果列表中
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])
    # 返回检测结果列表
    return detect_result


# 定义函数，接受一个参数：二维数组
def npmax(array):
    # 在数组的第二个维度（列）上找到最大值的索引
    arrayindex = array.argmax(1)
    # 在数组的第二个维度（列）上找到最大值
    arrayvalue = array.max(1)
    # 找到最大值数组中的最大值的索引，这个索引对应的是原数组的行索引
    i = arrayvalue.argmax()
    # 使用行索引i找到对应的列索引
    j = arrayindex[i]
    # 返回最大值的行索引和列索引
    return i, j
