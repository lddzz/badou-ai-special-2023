# 导入所需的库和模块
import cv2
import numpy as np
from keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model
import utils


# -----------------------------#
#   create_Pnet:创建PNet模型
#   粗略获取人脸框
#   输出bbox位置和是否有人脸
# -----------------------------#
# weight_path，这是预训练权重的文件路径
def create_Pnet(weight_path):
    # Step 1: 定义输入层
    # 输入层，图像的形状为[None, None, 3]
    input = Input(shape=[None, None, 3])

    # Step 2: 定义卷积层和激活函数
    # 第一层卷积
    # 使用10个3x3的卷积核,步长为1，不进行边界填充(padding='valid'),命名为conv1
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    # PReLU激活函数，共享轴为第1和第2维,命名为PReLU1
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    # 最大池化层，池化窗口大小为2x2
    x = MaxPool2D(pool_size=2)(x)

    # 第二层卷积
    # 使用16个3x3的卷积核,步长为1，不进行边界填充(padding='valid'),命名为conv2
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    # PReLU激活函数，共享轴为第1和第2维,命名为PReLU2
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)
    # 第三层卷积
    # 使用32个3x3的卷积核,步长为1，不进行边界填充(padding='valid'),命名为conv3
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    # PReLU激活函数，共享轴为第1和第2维
    x = PReLU(shared_axes=[1, 2])(x)

    # Step 3: 定义输出层
    # 分类层，2个1x1的卷积核，使用softmax激活函数，命名为'conv4-1',输出人脸/非人脸的概率
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 4个1x1的卷积核,无激活函数，线性，命名为'conv4-2'输出用于回归的人脸框坐标
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    # Step 4: 创建模型
    # 创建模型，输入为input，输出为分类和回归的结果
    model = Model([input], [classifier, bbox_regress])
    # Step 5: 加载预训练权重
    # 通过模型的load_weights方法加载预训练权重
    # weight_path是权重文件的路径
    # by_name=True表示按照层的名字加载权重
    model.load_weights(weight_path, by_name=True)
    # Step 6: 返回模型
    # 返回模型
    return model


# -----------------------------#
#   create_Rnet:创建RNet模型
#   mtcnn的第二段
#   精修框
# -----------------------------#
# weight_path，这是预训练权重的文件路径
def create_Rnet(weight_path):
    # Step 1: 定义输入层
    # 输入层，图像的形状为[24, 24, 3]
    input = Input(shape=[24, 24, 3])

    # Step 2: 定义卷积层和激活函数
    # 24,24,3 -> 11,11,28，第一层卷积
    # 使用28个3x3的卷积核,步长为1，不进行边界填充(padding='valid'),命名为conv1
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    # PReLU激活函数，共享轴为第1和第2维，命名为'prelu1'
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    # 最大池化层，池化窗口大小为3x3，步长为2，使用"same"填充
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # 11,11,28 -> 4,4,48，第二层卷积
    # 使用48个3x3的卷积核,步长为1，不进行边界填充(padding='valid'),命名为conv2
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    # PReLU激活函数，共享轴为第1和第2维，命名为'prelu2'
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    # 最大池化层，池化窗口大小为3x3，步长为2
    x = MaxPool2D(pool_size=3, strides=2)(x)
    # 4,4,48 -> 3,3,64，第三层卷积
    # 使用64个2x2的卷积核,步长为1，不进行边界填充(padding='valid'),命名为conv3
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    # PReLU激活函数，共享轴为第1和第2维，命名为'prelu3'
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # Step 3: 定义全连接层和激活函数
    # 3,3,64 -> 64,3,3
    # 使用Permute层进行轴置换，将原始输入的第3维移动到新的第1维，第2维移动到新的第2维，第1维移动到新的第3维
    x = Permute((3, 2, 1))(x)
    # 将三维的输入展平为一维
    x = Flatten()(x)
    # 576 -> 128，全连接层
    # 全连接层，将展平后的576维输入映射为128维输出,命名为'conv4'
    x = Dense(128, name='conv4')(x)
    # PReLU激活函数，命名为'prelu4'
    x = PReLU(name='prelu4')(x)

    # Step 4: 定义输出层
    # 128 -> 2，分类层，使用输出的维度为2,softmax激活函数，命名为'conv5-1'
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    # 128 -> 4，回归层，输出的维度为4,命名为'conv5-2'
    bbox_regress = Dense(4, name='conv5-2')(x)

    # Step 5: 创建模型
    # 创建模型，输入为input，输出为分类和回归的结果
    model = Model([input], [classifier, bbox_regress])

    # Step 6: 加载预训练权重
    # 通过模型的load_weights方法加载预训练权重
    # weight_path是权重文件的路径
    # by_name=True表示按照层的名字加载权重
    model.load_weights(weight_path, by_name=True)

    # Step 7: 返回模型
    # 返回构建好的RNet模型
    return model


# 创建ONet模型的函数
def create_Onet(weight_path):
    # Step 1: 定义输入层
    # 输入层，图像的形状为[48, 48, 3]
    input = Input(shape=[48, 48, 3])

    # Step 2: 定义卷积层和激活函数
    # 第一层卷积，输出形状为[46, 46, 32]
    # 使用32个3x3的卷积核,步长为1，不进行边界填充(padding='valid'),命名为conv1
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    # PReLU激活函数，共享轴为第1和第2维，命名为'prelu1'
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    # 最大池化层，池化窗口大小为3x3，步长为2，使用"same"填充
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # 第二层卷积，输出形状为[10, 10, 64]
    # 使用64个3x3的卷积核,步长为1，不进行边界填充(padding='valid'),命名为conv2
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    # PReLU激活函数，共享轴为第1和第2维，命名为'prelu2'
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    # 最大池化层，池化窗口大小为3x3，步长为2，
    x = MaxPool2D(pool_size=3, strides=2)(x)
    # 第三层卷积，输出形状为[4, 4, 64]
    # 使用64个3x3的卷积核,步长为1，不进行边界填充(padding='valid'),命名为conv3
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    # PReLU激活函数，共享轴为第1和第2维，命名为'prelu3'
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    # 最大池化层，池化窗口大小为2x2
    x = MaxPool2D(pool_size=2)(x)

    # Step 3: 定义全连接层和激活函数
    # 第四层卷积，输出形状为[3, 3, 128]
    # 使用128个2x2的卷积核,步长为1，不进行边界填充(padding='valid'),命名为conv4/
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    # PReLU激活函数，共享轴为第1和第2维，命名为'prelu4'
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)
    # # 使用Permute层进行轴置换，将原始输入的第3维移动到新的第1维，第2维移动到新的第2维，第1维移动到新的第3维
    x = Permute((3, 2, 1))(x)
    # 将三维的输入展平为一维，输出形状为[128*12*12]
    x = Flatten()(x)
    # 全连接层，输出形状为256,命名为'conv5'
    x = Dense(256, name='conv5')(x)
    # PReLU激活函数,命名为'prelu5'
    x = PReLU(name='prelu5')(x)

    # Step 4: 定义输出层
    # 分类层，输出形状为2，使用softmax激活函数,命名为'conv6-1'
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    # 回归层，输出形状为4,命名为'conv6-2'
    bbox_regress = Dense(4, name='conv6-2')(x)
    # 关键点回归层，输出形状为10,命名为'conv6-3'
    landmark_regress = Dense(10, name='conv6-3')(x)

    # Step 5: 创建模型
    # [input]是模型的输入
    # [classifier, bbox_regress, landmark_regress]是模型的输出
    # classifier是分类层，用于输出人脸/非人脸的概率。
    # bbox_regress是回归层，用于输出用于回归的人脸框坐标。
    # landmark_regress是关键点回归层，用于输出用于回归的人脸关键点坐标。
    model = Model([input], [classifier, bbox_regress, landmark_regress])

    # Step 6: 加载预训练权重
    # 通过模型的load_weights方法加载预训练权重，weight_path是权重文件的路径，
    # by_name=True表示按照层的名字加载权重
    model.load_weights(weight_path, by_name=True)

    # Step 7: 返回模型
    # 返回构建好的ONet模型
    return model


# 创建 MTCNN 类，用于人脸检测
class mtcnn:
    # 类的初始化函数,创建了PNet、RNet和ONet三个模型的实例,并加载了预训练的权重
    def __init__(self):
        # 创建 PNet 模型实例，并加载预训练权重
        self.Pnet = create_Pnet('model_data/pnet.h5')
        # 创建 RNet 模型实例，并加载预训练权重
        self.Rnet = create_Rnet('model_data/rnet.h5')
        # 创建 ONet 模型实例，并加载预训练权重
        self.Onet = create_Onet('model_data/onet.h5')

    # 用于执行人脸检测，通过调用PNet、RNet和ONet模型，返回检测到的人脸框列表
    def detectFace(self, img, threshold):
        # -----------------------------#
        #   # Step 1: 归一化输入图像并获取其尺寸,把[0,255]映射到(-1,1)
        # -----------------------------#
        # 从图像的每个像素中减去 127.5,将原始的范围从 [0, 255] 转换为 [-127.5, 127.5]
        # 将结果除以 127.5，将像素值缩放到范围 [-1, 1]
        copy_img = (img.copy() - 127.5) / 127.5
        # 获取归一化后图像的高度、宽度和通道数
        origin_h, origin_w, _ = copy_img.shape

        # -----------------------------#
        # Step 2: 计算图像金字塔的不同缩放比例
        # -----------------------------#
        # 计算图像金字塔的不同缩放比例
        scales = utils.calculateScales(img)

        # -----------------------------#
        # Step 3: 粗略计算人脸框:对每个缩放比例的图像使用PNet进行预测
        # -----------------------------#
        # 存储Pnet输出的结果
        out = []
        # 遍历图像金字塔的不同缩放比例
        for scale in scales:
            # 计算当前缩放比例下的图像高度
            hs = int(origin_h * scale)
            # 计算当前缩放比例下的图像宽度
            ws = int(origin_w * scale)
            # 缩放图像
            # 使用OpenCV的resize函数，将copy_img图像的大小调整为宽度为ws，高度为hs
            # 调整后的图像存储在scale_img中
            scale_img = cv2.resize(copy_img, (ws, hs))
            # 将缩放后的图像进行reshape，添加一个额外的维度，使其可以作为一个批量的一部分输入到模型中
            # 新的形状的第一个维度是1，其余的维度与scale_img的原始形状相同,inputs
            inputs = scale_img.reshape(1, *scale_img.shape)
            # 图像金字塔中的每张图片分别传入Pnet得到output
            output = self.Pnet.predict(inputs)
            # 将所有output加入out
            out.append(output)

        # -----------------------------#
        # Step 4: 对PNet的预测结果进行解码并进行非极大值抑制
        # -----------------------------#
        # 获取图像金字塔中的图像数量image_num
        image_num = len(scales)
        # 存储人脸框的列表rectangles
        rectangles = []
        # 遍历图像金字塔中的每张图像
        for i in range(image_num):
            # 提取当前缩放下Pnet输出的人脸概率图
            # 提取第i个缩放比例的图像的PNet输出结果中的人脸分类概率
            # out[i][0][0]是一个三维数组，表示PNet的输出结果
            # [:, :, 1]是一个切片操作，选择第三个维度的第二个元素，即人脸的概率
            # 将人脸概率图赋值给cls_prob
            cls_prob = out[i][0][0][:, :, 1]
            # 提取第i个缩放比例的图像的PNet输出结果中的人脸框位置信息
            # out[i][1][0]是一个三维数组，表示PNet的输出结果
            # 这个数组的第三个维度的大小为4，表示人脸框的位置信息（左上角的x和y坐标，右下角的x和y坐标）
            # 将人脸框位置信息赋值给roi
            roi = out[i][1][0]
            # 取出每个缩放后图片的长宽
            # 获取当前缩放下人脸概率图的高度和宽度
            out_h, out_w = cls_prob.shape
            # 获取人脸概率图的最大边长
            out_side = max(out_h, out_w)
            # 打印当前缩放下人脸概率图的形状
            print(f"当前缩放下人脸概率图的形状为：{cls_prob.shape}")
            # 解码过程
            # 调用工具函数detect_face_12net进行人脸检测
            # 传入当前缩放下的人脸概率图(cls_prob)、对应的框的位置信息(roi)、
            # 人脸概率图的最大边长(out_side)、缩放比例的倒数(1 / scales[i])、
            # 原始图像的宽度(origin_w)、原始图像的高度(origin_h)、
            # 人脸检测的阈值(threshold[0])
            # 返回的是检测到的人脸框的列表
            rectangle = utils.detect_face_12net(
                cls_prob,
                roi,
                out_side,
                1 / scales[i],
                origin_w,
                origin_h,
                threshold[0]
            )
            # 将检测到的人脸框列表加入总的人脸框列表(rectangles)中
            rectangles.extend(rectangle)
        # 进行非极大抑制
        # 使用非极大抑制(NMS)处理人脸框，去除重叠度较高的框，保留置信度较高的框
        # 传入人脸框列表(rectangles)和NMS的阈值(0.7)
        rectangles = utils.NMS(rectangles, 0.7)

        # -----------------------------#
        # Step 5: 稍微精确计算人脸框:对每个人脸框使用RNet进行预测并筛选
        # -----------------------------#
        # 如果经过NMS处理后的人脸框数量为0，直接返回空的人脸框列表
        if len(rectangles) == 0:
            return rectangles
        # 存储经过NMS处理后的人脸框裁剪并缩放后的图像批量predict_24_batch
        predict_24_batch = []
        # 遍历经过NMS处理后的人脸框
        for rectangle in rectangles:
            # 根据当前人脸框的坐标在原始图像中裁剪出人脸图像
            # rectangle[1]和rectangle[3]是人脸框的y坐标，对应图像的高度
            # rectangle[0]和rectangle[2]是人脸框的x坐标，对应图像的宽度
            # 裁剪出的人脸图像存储在crop_img中
            crop_img = copy_img[
                       int(rectangle[1]):int(rectangle[3]),
                       int(rectangle[0]):int(rectangle[2])
                       ]
            # 缩放裁剪出的人脸图像到24x24大小scale_img
            scale_img = cv2.resize(crop_img, (24, 24))
            # 将缩放后的人脸图像添加到24x24大小的图像批量(predict_24_batch)中
            predict_24_batch.append(scale_img)
        # 将24x24大小的人脸图像批量转换为NumPy数组
        predict_24_batch = np.array(predict_24_batch)
        # 使用Rnet对24x24大小的人脸图像批量进行预测，得到Rnet的输出结果out
        out = self.Rnet.predict(predict_24_batch)
        # 提取Rnet的输出结果中的人脸分类概率cls_prob
        cls_prob = out[0]
        # 将人脸分类概率转换为NumPy数组
        cls_prob = np.array(cls_prob)
        # 提取Rnet的输出结果中的人脸框位置信息roi_prob
        roi_prob = out[1]
        # 将人脸框位置信息转换为NumPy数组
        roi_prob = np.array(roi_prob)
        # 使用工具函数filter_face_24net对Rnet的输出进行处理，得到经过Rnet筛选后的人脸框列表
        # cls_prob是RNet输出的人脸分类概率
        # roi_prob是RNet输出的人脸框位置信息
        # rectangles是经过PNet处理后的人脸框列表
        # origin_w和origin_h分别是原始图像的宽度和高度
        # threshold[1]是人脸检测的阈值
        # 函数的返回结果是经过RNet筛选后的人脸框列表，这个结果被赋值给rectangles
        rectangles = utils.filter_face_24net(
            cls_prob,
            roi_prob,
            rectangles,
            origin_w,
            origin_h,
            threshold[1]
        )

        # -----------------------------#
        # Step 6: 计算人脸框:对每个人脸框使用ONet进行预测并筛选
        # -----------------------------#
        # 如果经过Rnet筛选后的人脸框数量为0，直接返回空的人脸框列表。
        if len(rectangles) == 0:
            return rectangles

        # 存储经过Rnet筛选后的人脸框裁剪并缩放后的图像批量predict_48_batch
        predict_48_batch = []
        # 遍历经过Rnet筛选后的人脸框
        for rectangle in rectangles:
            # 根据当前人脸框的坐标在原始图像中裁剪出人脸图像
            # rectangle[1]和rectangle[3]是人脸框的y坐标，对应图像的高度
            # rectangle[0]和rectangle[2]是人脸框的x坐标，对应图像的宽度
            # 裁剪出的人脸图像存储在crop_img中
            crop_img = copy_img[
                       int(rectangle[1]):int(rectangle[3]),
                       int(rectangle[0]):int(rectangle[2])
                       ]

            # 缩放裁剪出的人脸图像到48x48大小scale_img
            scale_img = cv2.resize(crop_img, (48, 48))
            # 将缩放后的人脸图像添加到48x48大小的图像批量(predict_48_batch)中
            predict_48_batch.append(scale_img)
        # 将48x48大小的人脸图像批量转换为NumPy数组
        predict_48_batch = np.array(predict_48_batch)
        # 使用Onet对48x48大小的人脸图像批量进行预测，得到Onet的输出结果output
        output = self.Onet.predict(predict_48_batch)
        # 提取Onet的输出结果中的人脸分类概率cls_prob
        cls_prob = output[0]
        # 提取Onet的输出结果中的人脸框位置信息roi_prob
        roi_prob = output[1]
        # 提取Onet的输出结果中的关键点位置信息pts_prob
        pts_prob = output[2]
        # 使用工具函数filter_face_48net对Onet的输出进行处理，得到经过Onet筛选后的人脸框列表
        # 传入Onet输出的人脸分类概率(cls_prob)
        # 人脸框位置信息(roi_prob)
        # 关键点位置信息(pts_prob)
        # 经过Rnet筛选后的人脸框列表(rectangles)
        # 原始图像的宽度(origin_w)
        # 原始图像的高度(origin_h)
        # 人脸检测的阈值(threshold[2])
        # 返回的是经过Onet筛选后的人脸框列表rectangles
        rectangles = utils.filter_face_48net(
            cls_prob,
            roi_prob,
            pts_prob,
            rectangles,
            origin_w,
            origin_h,
            threshold[2]
        )
        # -----------------------------#
        # Step 7: 返回最终的人脸框列表
        # -----------------------------#
        # 返回最终的人脸框列表
        return rectangles
