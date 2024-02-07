# 导入所需的库
import cv2  # OpenCV库,用于图像处理
from mtcnn import mtcnn  # 导入MTCNN库,用于人脸检测

# 读取图像文件img/timg.jpg
img = cv2.imread("img/timg.jpg")
# 创建MTCNN模型实例
model = mtcnn()
# 设置三个不同网络的置信度阈值threshold:[0.5, 0.6, 0.7]
threshold = [0.5, 0.6, 0.7]
# 利用MTCNN模型detectFace方法检测图像中的人脸,返回矩形框的坐标信息rectangles
rectangles = model.detectFace(img, threshold)
# 创建原始图像的副本draw,用于在其上绘制检测结果
draw = img.copy()
# 遍历检测到的人脸矩形框
for rectangle in rectangles:
    # 检查矩形框是否存在
    if rectangle is not None:
        # 计算矩形的宽度(W)
        # rectangle[0] 表示矩形左上角的 x 坐标
        # rectangle[2] 表示矩形右下角的 x 坐标
        # 计算矩形的宽度(W)，即右下角 x 坐标减去左上角 x 坐标的差值
        W = int(rectangle[2] - rectangle[0])

        # 计算矩形的高度(H)
        # rectangle[1] 表示矩形左上角的 y 坐标
        # rectangle[3] 表示矩形右下角的 y 坐标
        # 计算矩形的高度(H)，即右下角 y 坐标减去左上角 y 坐标的差值
        H = int(rectangle[3] - rectangle[1])

        # 计算填充值paddingH,其中0.01 * W用于计算高度的1 % 填充,
        paddingH = 0.01 * W
        # 计算填充值paddingW,0.02 * H用于计算宽度的2 % 填充
        paddingW = 0.02 * H
        # 裁剪出人脸区域crop_img
        # rectangle[1] 表示矩形左上角的 y 坐标
        # rectangle[3] 表示矩形右下角的 y 坐标
        # rectangle[0] 表示矩形左上角的 x 坐标
        # rectangle[2] 表示矩形右下角的 x 坐标
        # paddingH 是用于计算高度的填充值
        # paddingW 是用于计算宽度的填充值
        # int(rectangle[1] + paddingH):int(rectangle[3] - paddingH) 表示裁剪的垂直范围
        # 从矩形的上边界开始，向下裁剪 paddingH 的高度，然后向上裁剪 paddingH 的高度
        # int(rectangle[0] - paddingW):int(rectangle[2] + paddingW) 表示裁剪的水平范围
        # 从矩形的左边界开始，向右裁剪 paddingW 的宽度，然后向左裁剪 paddingW 的宽度
        crop_img = img[
                   int(rectangle[1] + paddingH):int(rectangle[3] + paddingH),
                   int(rectangle[0] + paddingW):int(rectangle[2] + paddingW)
                   ]

        # 如果裁剪后的图像为空,则继续下一次循环
        if crop_img is None:
            continue
        # 如果裁剪后的图像宽度或高度小于零,则继续下一次循环
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        # 绘制人脸矩形框
        # 在图像上绘制人脸矩形框
        # draw 是原始图像的副本，用于在其上绘制检测结果
        # (int(rectangle[0]), int(rectangle[1])) 是矩形的左上角坐标
        # (int(rectangle[2]), int(rectangle[3])) 是矩形的右下角坐标
        # color=(255, 0, 0) 指定绘制的矩形框的颜色，这里是蓝色，格式为 (B, G, R)
        # thickness=1 指定绘制的矩形框的线宽度
        cv2.rectangle(
            draw,
            pt1=(int(rectangle[0]), int(rectangle[1])),
            pt2=(int(rectangle[2]), int(rectangle[3])),
            color=(0, 0, 0),
            thickness=2
        )

        # 绘制人脸关键点
        # 遍历人脸关键点的坐标（从5到15，步长为2）
        for i in range(5, 15, 2):
            # rectangle[i] 表示关键点的 x 坐标
            # rectangle[i + 1] 表示关键点的 y 坐标
            # center=(int(rectangle[i]), int(rectangle[i + 1])) 是关键点的坐标
            # radius=2 指定绘制的圆圈的半径
            # color=(0, 255, 0) 指定绘制的圆圈的颜色，这里是绿色，格式为 (B, G, R)
            cv2.circle(
                draw,
                center=(int(rectangle[i]), int(rectangle[i + 1])),
                radius=2,
                color=(0, 255, 0)
            )
# 将绘制了矩形框和关键点的图像保存到文件系统中
cv2.imwrite("img/out.jpg", draw)
# 展示绘制了检测结果的图像,并等待用户按下任意键后关闭窗口
cv2.imshow("test", draw)
# 等待用户按下任意键
cv2.waitKey(0)
# 使用OpenCV的destroyAllWindows函数关闭所有图像窗口
cv2.destroyAllWindows()
