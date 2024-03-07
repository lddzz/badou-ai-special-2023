import numpy as np


# -----------------------------#
#   计算原始输入图像
#   每一次缩放的比例
# -----------------------------#
# calculateScales:
# 传入img
# 返回不同缩放比例的结果列表scales
def calculateScales(img):
    # 复制图像
    copy_img = img.copy()
    # 初始缩放比例pr_scale为1.0
    pr_scale = 1.0
    # 获取图像的高度h,宽度w,通道数_
    h, w, _ = copy_img.shape
    # 计算优化项的伸缩比例，即将图像的最小边伸缩至500像素，同时保持长宽比不变
    # 判断图像的最小边长是否大于500像素
    if min(h, w) > 500:
        # 计算优化项的伸缩比例，将图像的最小边缩放至500像素
        pr_scale = 500.0 / min(h, w)
        # 缩放图像宽度
        h = int(h * pr_scale)
        # 缩放图像高度
        w = int(w * pr_scale)
    # 如果图像的最大边长小于500像素
    elif max(h, w) < 500:
        # 计算优化项的伸缩比例，将图像的最大边缩放至500像素
        pr_scale = 500.0 / max(h, w)
        # 缩放图像宽度
        h = int(h * pr_scale)
        # 缩放图像高度
        w = int(w * pr_scale)
    # 用于存储不同缩放比例的结果scales
    scales = []
    # 缩放比例因子factor为0.709
    factor = 0.709
    # 缩放比例因子计数factor_count为0
    factor_count = 0
    # 获取图像的最小边长
    minl = min(h, w)
    # 根据图像的最小边长，计算不同的缩放比例，并将其添加到结果列表中
    # while 循环，生成不同缩放比例的图像尺度，以确保多尺度人脸检测
    while minl >= 12:
        # 将当前计算得到的缩放比例添加到列表中
        # 参数说明：
        # pr_scale: 用于图像缩放的基准比例，确保图像能够适应不同尺度的人脸。
        # factor: 缩放比例的因子，控制不同尺度之间的缩放关系，约为0.709。
        # factor_count: 计数器，表示当前缩放比例的次方，用于生成一系列递减的缩放比例。
        # pow(factor, factor_count): 缩放因子的指数部分，用于调整缩放比例的大小。
        # pr_scale * pow(factor, factor_count): 当前迭代下的图像缩放比例。
        # scales: 存储不同缩放比例的列表，用于多尺度人脸检测。
        scales.append(pr_scale * pow(factor, factor_count))
        # 将当前图像的最小边长乘以缩放因子，用于下一次迭代计算
        minl *= factor
        # 递增缩放因子的指数，用于计算下一次的缩放比例
        factor_count += 1
    # 返回不同缩放比例的结果列表
    return scales


# -----------------------------#
#   将长方形调整为正方形
# -----------------------------#
# rect2square:将长方形调整为正方形
# 传参:rectangles矩形框
# 返回:rectangles矩形框
def rect2square(rectangles):
    # 计算矩形的宽度w，通过从矩形右下角的x坐标减去左上角的x坐标
    w = rectangles[:, 2] - rectangles[:, 0]
    # 计算矩形的高度h，通过从矩形右下角的y坐标减去左上角的y坐标
    h = rectangles[:, 3] - rectangles[:, 1]
    # 计算宽度和高度的最大值，并将结果进行转置。这个值将用于确定正方形的边长
    l = np.maximum(w, h).T
    # 更新左上角点的x坐标
    # rectangles[:, 0]：表示选取矩形数组中所有行的第一列，即左上角点的 x 坐标
    # w * 0.5：计算每个矩形宽度的一半
    # l * 0.5：计算正方形边长的一半
    # rectangles[:, 0] + w * 0.5 - l * 0.5：这一部分计算的是新的 x 坐标
    # 首先，从原始 x 坐标开始 (rectangles[:, 0])
    # 然后加上矩形宽度的一半 (w * 0.5)，最后减去正方形边长的一半 (l * 0.5)
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    # 更新左上角点的y坐标
    # rectangles[:, 1]：选择矩形数组中所有行的第二列，表示每个矩形的左上角点的 y 坐标
    # h * 0.5：计算每个矩形高度的一半
    # l * 0.5：计算正方形边长的一半
    # rectangles[:, 1] + h * 0.5 - l * 0.5：原始值加上高度的一半，减去正方形边长的一半
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    # 更新右下角点的坐标
    # rectangles[:, 2:4]：选择矩形数组中所有行的第三和第四列，表示每个矩形的右下角点的 x 和 y 坐标。
    # rectangles[:, 0:2]：选择矩形数组中所有行的第一和第二列，表示每个矩形的左上角点的 x 和 y 坐标。
    # np.repeat([l], 2, axis=0)：创建一个形状为 (2, n) 的数组，其中 n 是矩形数量，这个数组包含正方形边长的一半。np.repeat 用于复制这个数组，确保它有两行（对应 x 和 y 坐标）。
    # .T：对这个数组进行转置，以便在矩形数组中正确地与左上角点的坐标相加。
    # rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T：将左上角点的坐标与正方形边长的一半相加，得到右下角点的坐标。
    # rectangles[:, 2:4] = ...：将右下角点的坐标赋值给矩形数组中对应的列，从而更新每个矩形的右下角点
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    # 返回更新后的矩形框
    return rectangles


# -------------------------------------#
#   非极大抑制
# -------------------------------------#
# NMS: 这是函数的名称，代表非极大值抑制（NMS）。
# rectangles: 是一个输入参数，代表矩形框的列表。
# threshold: 是另一个输入参数，代表NMS中用于过滤重叠框的阈值
def NMS(rectangles, threshold):
    # 如果矩形框列表为空，则直接返回列表
    if len(rectangles) == 0:
        return rectangles
    # 将矩形框列表转换为numpy数组boxes
    boxes = np.array(rectangles)
    # 提取矩形框的左上角和右下角坐标以及目标尺寸
    # x1: 提取了矩形框的左上角 x 坐标。
    x1 = boxes[:, 0]
    # y1: 提取了矩形框的左上角 y 坐标。
    y1 = boxes[:, 1]
    # x2: 提取了矩形框的右下角 x 坐标。
    x2 = boxes[:, 2]
    # y2: 提取了矩形框的右下角 y 坐标。
    y2 = boxes[:, 3]
    # sc: 提取了矩形框的概率得分（score）
    sc = boxes[:, 4]
    # 计算矩形框的面积
    # x2 - x1 + 1: 计算每个矩形框在 x 轴上的宽度（加1是为了避免宽度为零的情况）。
    # y2 - y1 + 1: 计算每个矩形框在 y 轴上的高度（同样加1是为了避免高度为零的情况）
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    # 根据目标尺寸对矩形框进行排序
    I = np.array(sc.argsort())
    # 创建空列表pick，用于存储最终被选中的矩形框的索引
    pick = []
    # 循环直到矩形框列表为空
    while len(I) > 0:
        # 提取最高概率得分的矩形框和其余矩形框的左上角和右下角坐标
        # xx1: 计算了重叠矩形框左上角 x 坐标的最大值。
        # x1[I[-1]]: 表示最高概率得分的矩形框在 x 轴上的左上角坐标 x1 的取值。
        # x1[I[0:-1]]: 表示除了最高概率得分的矩形框之外的所有其他矩形框在 x 轴上的左上角坐标 x1 的取值
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        # yy1: 计算了重叠矩形框左上角 y 坐标的最大值。
        # y1[I[-1]]: 表示最高概率得分的矩形框在 y 轴上的左上角坐标 y1 的取值。
        # y1[I[0:-1]]: 表示除了最高概率得分的矩形框之外的所有其他矩形框在 y 轴上的左上角坐标 y1 的取值
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        # xx2: 计算了重叠矩形框右下角 x 坐标的最小值
        # x2[I[-1]]: 表示最高概率得分的矩形框在 x 轴上的右下角坐标 x2 的取值。
        # x2[I[0:-1]]: 表示除了最高概率得分的矩形框之外的所有其他矩形框在 x 轴上的右下角坐标 x2 的取值
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        # yy2: 计算了重叠矩形框右下角 y 坐标的最小值
        # y2[I[-1]]: 表示最高概率得分的矩形框在 y 轴上的右下角坐标 y2 的取值。
        # y2[I[0:-1]]: 表示除了最高概率得分的矩形框之外的所有其他矩形框在 y 轴上的右下角坐标 y2 的取值
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        # 计算重叠部分的宽度和高度
        # xx2 - xx1 + 1: 计算了重叠矩形框在 x 轴上的宽度。
        w = np.maximum(0.0, xx2 - xx1 + 1)
        # yy2 - yy1 + 1: 计算了重叠矩形框在 y 轴上的高度
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # 计算重叠部分的面积
        inter = w * h
        # 计算重叠面积占比
        # area[I[-1]]: 表示最高概率得分的矩形框的面积。
        # area[I[0:-1]]: 表示除了最高概率得分的矩形框之外的所有其他矩形框的面积。
        # inter: 表示重叠部分的面积
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        # 将最高概率得分的矩形框添加到结果列表中
        pick.append(I[-1])
        # 根据阈值删除重叠面积占比大于阈值的矩形框
        I = I[np.where(o <= threshold)[0]]

    # boxes[pick]: 从原始矩形框数组 boxes 中选择被选中的索引 pick 所对应的矩形框，创建一个新的NumPy数组。
    # .tolist(): 将新的NumPy数组转换为Python列表
    result_rectangle = boxes[pick].tolist()
    # 返回结果矩形框列表
    return result_rectangle


# -------------------------------------#
#   对pnet处理后的结果进行处理
# -------------------------------------#
# detect_face_12net:通过12x12网络（P-Net）处理后的人脸检测，返回经非极大值抑制（NMS）过滤后的人脸框
# - cls_prob:形状为 (H, W)，表示分类概率的数组，H和W为高度和宽度
# - roi:形状为 (4, H, W)，表示边界框回归信息的数组
# - out_side:输出图像边长
# - scale:缩放因子
# - width:输入图像宽度
# - height:输入图像高度
# - threshold:用于分类概率的阈值
# 返回:
# List, 包含检测到的人脸边界框的列表，每个边界框用 [x1, y1, x2, y2, score] 表示
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    # 将分类概率矩阵的行和列进行交换
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    # 将边界框回归信息矩阵的第一个维度（4）和第二维度进行交换
    roi = np.swapaxes(roi, 0, 2)
    # 计算步长
    stride = 0
    # 如果输出边长不为1
    if out_side != 1:
        # 计算步长(2 * out_side - 1) / (out_side - 1)
        stride = float(2 * out_side - 1) / (out_side - 1)
    # 找到满足阈值条件的分类概率的坐标
    (x, y) = np.where(cls_prob >= threshold)
    # 创建包含坐标对的二维数组boundingbox
    boundingbox = np.array([x, y]).T
    # 计算边界框坐标
    # 使用步长和缩放因子计算两个边界框的坐标
    # - boundingbox: 包含坐标对的二维数组，表示检测到的分类概率的坐标
    # - stride: 步长，用于计算边界框坐标
    # - scale: 缩放因子，用于调整边界框的大小
    # - bb1 和 bb2: 分别是两个边界框的坐标
    # 0 和 11 是用于计算偏移量的常数
    bb1 = np.fix((stride * boundingbox + 0) * scale)
    bb2 = np.fix((stride * boundingbox + 11) * scale)
    # 使用 NumPy 的 concatenate 函数将两个边界框坐标数组在列方向进行拼接
    # - bb1 和 bb2: 分别是两个边界框的坐标数组
    # - axis=1: 指定沿着列的方向进行拼接，即将两个数组的列连接在一起
    boundingbox = np.concatenate((bb1, bb2), axis=1)
    # 获取边界框的偏移量和得分信息
    # 获取左上角 x 坐标的偏移量
    dx1 = roi[0][x, y]
    # 获取右上角 y 坐标的偏移量
    dx2 = roi[1][x, y]
    # 获取左下角 x 坐标的偏移量
    dx3 = roi[2][x, y]
    # 获取右下角 y 坐标的偏移量
    dx4 = roi[3][x, y]
    # 获取分类得分，并将其转置成列向量
    # - cls_prob: 形状为 (H, W) 的数组，表示分类概率的矩阵。
    # - x 和 y: 分别是满足条件的分类概率矩阵元素的行和列的坐标。
    # - np.array([cls_prob[x, y]]): 通过索引操作获取分类概率矩阵中指定坐标的元素，并将其封装为一个数组。
    # - .T: 对数组进行转置，将其从行向量转换为列向量。
    # - score: 包含分类得分的数组，转置后成为列向量
    score = np.array([cls_prob[x, y]]).T
    # 获取四个偏移量，并将其转置成列向量
    # - dx1, dx2, dx3, dx4: 分别是四个偏移量的值，表示边界框的左上、右上、左下和右下角相对于原始边界框的偏移。
    # - np.array([dx1, dx2, dx3, dx4]): 通过索引操作获取偏移量的数组，并将其封装为一个数组。
    # - .T: 对数组进行转置，将其从行向量转换为列向量。
    # - offset: 包含四个偏移量的数组，转置后成为列向量
    offset = np.array([dx1, dx2, dx3, dx4]).T
    # 应用偏移量，调整边界框大小
    # - boundingbox: 包含坐标对的二维数组，表示检测到的分类概率的坐标
    # - offset: 包含四个偏移量的数组，用于调整边界框的位置
    # - 12.0: 常数，用于调整偏移量的影响
    # - scale: 缩放因子，用于调整边界框的大小
    boundingbox = boundingbox + offset * 12.0 * scale
    # 将边界框坐标boundingbox和得分score拼接成数组
    # axis=1: 指定沿着列的方向进行拼接，即将两个数组的列连接在一起
    rectangles = np.concatenate((boundingbox, score), axis=1)
    # 调用rect2square方法将矩形边界框rectangles转换为正方形边界框
    rectangles = rect2square(rectangles)
    # 对边界框进行处理，确保在图像范围内
    # 初始化一个空列表pick，用于存储处理后的边界框
    pick = []
    # 遍历之前计算得到的边界框 rectangles
    for i in range(len(rectangles)):
        # 获取左上角 x 坐标，确保不小于0
        x1 = int(max(0, rectangles[i][0]))
        # 获取左上角 y 坐标，确保不小于0
        y1 = int(max(0, rectangles[i][1]))
        # 获取右下角 x 坐标，确保不超过图像宽度
        x2 = int(min(width, rectangles[i][2]))
        # 获取右下角 y 坐标，确保不超过图像高度
        y2 = int(min(height, rectangles[i][3]))
        # 获取边界框的得分
        sc = rectangles[i][4]
        # 确保边界框在图像范围内且有意义
        if x2 > x1 and y2 > y1:
            # 将处理后的边界框坐标和得分添加到 pick 列表中
            pick.append([x1, y1, x2, y2, sc])
    # 调用非极大值抑制函数，去除重叠的边界框
    return NMS(pick, 0.3)


# -------------------------------------#
#   对Rnet处理后的结果进行处理
# -------------------------------------#
# filter_face_24net
# - cls_prob:分类概率
# - roi:感兴趣区域
# - rectangles:矩形框列表
# - width:图像宽度
# - height:图像高度
# - threshold:阈值
# - prob = cls_prob[:, 1]
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    # 提取分类概率中人脸的概率
    prob = cls_prob[:, 1]
    # 找出概率大于阈值的索引
    pick = np.where(prob >= threshold)
    # 将矩形框列表转换为numpy数组
    rectangles = np.array(rectangles)
    # 提取筛选后的矩形框的左上角和右下角坐标
    # 提取筛选后的矩形框的左上角x坐标
    x1 = rectangles[pick, 0]
    # 提取筛选后的矩形框的左上角y坐标
    y1 = rectangles[pick, 1]
    # 提取筛选后的矩形框的右下角x坐标
    x2 = rectangles[pick, 2]
    # 提取筛选后的矩形框的右下角y坐标
    y2 = rectangles[pick, 3]
    # 提取筛选后的对应概率值，并转置
    sc = np.array([prob[pick]]).T
    # 提取筛选后的roi偏移量
    # 提取筛选后的矩形框的左上角x坐标的偏移量
    dx1 = roi[pick, 0]
    # 提取筛选后的矩形框的左上角y坐标的偏移量
    dx2 = roi[pick, 1]
    # 提取筛选后的矩形框的右下角x坐标的偏移量
    dx3 = roi[pick, 2]
    # 提取筛选后的矩形框的右下角y坐标的偏移量
    dx4 = roi[pick, 3]
    # 计算筛选后的矩形框的宽和高
    w = x2 - x1
    h = y2 - y1
    # 根据roi偏移量调整矩形框的坐标，并转置
    # 提取筛选后的矩形框的左上角x坐标，并根据对应的roi偏移量和矩形框的宽度进行调整，然后转置
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    # 提取筛选后的矩形框的左上角y坐标，并根据对应的roi偏移量和矩形框的高度进行调整，然后转置
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    # 提取筛选后的矩形框的右下角x坐标，并根据对应的roi偏移量和矩形框的宽度进行调整，然后转置
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    # 提取筛选后的矩形框的右下角y坐标，并根据对应的roi偏移量和矩形框的高度进行调整，然后转置
    y2 = np.array([(y2 + dx4 * h)[0]]).T
    # 将新坐标和概率合并成一个新的矩形框数组
    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    # 将矩形框转换为正方形
    rectangles = rect2square(rectangles)
    # 初始化一个用于存储有效人脸矩形框的列表
    pick = []
    # 对每个筛选后的矩形框进行边界处理，并添加到pick列表中
    # 遍历矩形框列表
    for i in range(len(rectangles)):
        # 获取矩形框左上角x坐标，确保不小于0
        x1 = int(max(0, rectangles[i][0]))
        # 获取矩形框左上角y坐标，确保不小于0
        y1 = int(max(0, rectangles[i][1]))
        # 获取矩形框右下角x坐标，确保不超过图像宽度
        x2 = int(min(width, rectangles[i][2]))
        # 获取矩形框右下角y坐标，确保不超过图像高度
        y2 = int(min(height, rectangles[i][3]))
        # 获取矩形框的得分
        sc = rectangles[i][4]
        # 如果矩形框是有效的（即宽度和高度都大于0）
        if x2 > x1 and y2 > y1:
            # 将矩形框添加到pick列表中
            pick.append([x1, y1, x2, y2, sc])
    # 使用NMS算法进一步过滤矩形框并返回结果
    return NMS(pick, 0.3)


# -------------------------------------#
#   对onet处理后的结果进行处理
# -------------------------------------#
# filter_face_48net
# 参数：
# cls_prob：分类概率数组
# roi：roi数组
# pts：pts数组
# rectangles：矩形框数组
# width：图像宽度
# height：图像高度
# threshold：概率阈值
# 返回值：
# 过滤后的人脸矩形框数组
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    # 提取分类概率中人脸的概率
    prob = cls_prob[:, 1]
    # 找出概率大于阈值的索引
    pick = np.where(prob >= threshold)
    # 将矩形框列表转换为numpy数组
    rectangles = np.array(rectangles)
    # 提取筛选后的矩形框的左上角和右下角坐标
    # 从筛选后的矩形框数组中提取出每个矩形框的左上角的x坐标
    x1 = rectangles[pick, 0]
    # 从筛选后的矩形框数组中提取出每个矩形框的左上角的y坐标
    y1 = rectangles[pick, 1]
    # 从筛选后的矩形框数组中提取出每个矩形框的右下角的x坐标
    x2 = rectangles[pick, 2]
    # 从筛选后的矩形框数组中提取出每个矩形框的右下角的y坐标
    y2 = rectangles[pick, 3]
    # 提取筛选后的对应概率值，并转置
    sc = np.array([prob[pick]]).T
    # 提取筛选后的roi偏移量
    # 从筛选后的roi数组中提取出每个矩形框的左上角x坐标的偏移量
    dx1 = roi[pick, 0]
    # 从筛选后的roi数组中提取出每个矩形框的左上角y坐标的偏移量
    dx2 = roi[pick, 1]
    # 从筛选后的roi数组中提取出每个矩形框的右下角x坐标的偏移量
    dx3 = roi[pick, 2]
    # 从筛选后的roi数组中提取出每个矩形框的右下角y坐标的偏移量
    dx4 = roi[pick, 3]
    # 计算筛选后的矩形框的宽和高
    w = x2 - x1
    h = y2 - y1
    # 根据roi偏移量和矩形框的宽度，调整特征点0的x坐标，并转置
    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    # 根据roi偏移量和矩形框的高度，调整特征点1的y坐标，并转置
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    # 根据roi偏移量和矩形框的宽度，调整特征点2的x坐标，并转置
    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    # 根据roi偏移量和矩形框的高度，调整特征点3的y坐标，并转置
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
    # 根据roi偏移量和矩形框的宽度，调整特征点4的x坐标，并转置
    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    # 根据roi偏移量和矩形框的高度，调整特征点5的y坐标，并转置
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    # 根据roi偏移量和矩形框的宽度，调整特征点6的x坐标，并转置
    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    # 根据roi偏移量和矩形框的高度，调整特征点7的y坐标，并转置
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
    # 根据roi偏移量和矩形框的宽度，调整特征点8的x坐标，并转置
    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    # 根据roi偏移量和矩形框的高度，调整特征点9的y坐标，并转置
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

    # 根据roi偏移量调整矩形框的坐标，并转置
    # 根据roi偏移量和矩形框的宽度，调整矩形框的左上角x坐标，并转置
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    # 根据roi偏移量和矩形框的高度，调整矩形框的左上角y坐标，并转置
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    # 根据roi偏移量和矩形框的宽度，调整矩形框的右下角x坐标，并转置
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    # 根据roi偏移量和矩形框的高度，调整矩形框的右下角y坐标，并转置
    y2 = np.array([(y2 + dx4 * h)[0]]).T
    # 将新坐标、概率、特征点坐标合并成一个新的矩形框数组
    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)
    # 初始化一个用于存储有效人脸矩形框的列表
    pick = []
    # 对每个筛选后的矩形框进行边界处理，并添加到pick列表中
    # 遍历矩形框列表
    for i in range(len(rectangles)):
        # 获取矩形框左上角x坐标，确保不小于0
        x1 = int(max(0, rectangles[i][0]))
        # 获取矩形框左上角y坐标，确保不小于0
        y1 = int(max(0, rectangles[i][1]))
        # 获取矩形框右下角x坐标，确保不超过图像宽度
        x2 = int(min(width, rectangles[i][2]))
        # 获取矩形框右下角y坐标，确保不超过图像高度
        y2 = int(min(height, rectangles[i][3]))
        # 如果矩形框是有效的（即宽度和高度都大于0）
        if x2 > x1 and y2 > y1:
            # 将矩形框添加到pick列表中
            # 矩形框的左上角x坐标
            # 矩形框的左上角y坐标
            # 矩形框的右下角x坐标
            # 矩形框的右下角y坐标
            # 矩形框的概率得分
            # 矩形框的10个特征点的坐标
            pick.append([x1, y1, x2, y2, rectangles[i][4], rectangles[i][5], rectangles[i][6], rectangles[i][7],
                         rectangles[i][8], rectangles[i][9], rectangles[i][10], rectangles[i][11], rectangles[i][12],
                         rectangles[i][13], rectangles[i][14]])
    # 使用NMS算法进一步过滤矩形框并返回结果
    return NMS(pick, 0.3)
