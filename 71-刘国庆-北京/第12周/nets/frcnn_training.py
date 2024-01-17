# 导入图像预处理函数
from keras.applications.imagenet_utils import preprocess_input
# 导入Keras后端接口
from keras import backend as K
# 导入Keras和TensorFlow库
import keras
import tensorflow as tf
# 导入NumPy库
import numpy as np
# 导入Python内置的伪随机数生成模块
from random import shuffle
# 导入Python内置的随机数生成模块
import random
# 导入图像处理模块
from PIL import Image
# 导入Keras中的多类别分类问题损失函数
from keras.objectives import categorical_crossentropy
# 导入Matplotlib中颜色空间转换函数
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
# 导入自定义的anchors模块中的获取anchors函数
from utils.anchors import get_anchors


# 定义一个生成指定范围内随机数的函数，参数a和b为范围
def rand(a=0, b=1):
    # 生成随机数并将其映射到指定范围[a, b)
    return np.random.rand() * (b - a) + a


# 定义分类损失函数，带有可调整的正负样本权重比例
def cls_loss(ratio=3):
    # 定义具体的分类损失计算函数 _cls_loss，接受真实标签y_true和预测结果y_pred
    def _cls_loss(y_true, y_pred):
        # 将输入中的真实标签赋值给变量 labels
        labels = y_true
        # 通过切片操作，提取每个先验框的状态信息,-1 表示需要忽略的先验框，0 表示背景，1 表示存在目标
        anchor_state = y_true[:, :, -1]
        # 将模型的预测结果赋值给变量 classification，表示每个先验框属于不同类别的概率
        classification = y_pred
        # 找出标记为存在目标（状态为1）的先验框的索引
        indices_for_object = tf.where(keras.backend.equal(anchor_state, 1))
        # 使用 tf.gather_nd 函数根据找到的存在目标的先验框的索引从 labels 中提取对应的真实标签
        labels_for_object = tf.gather_nd(labels, indices_for_object)
        # 使用 tf.gather_nd 函数根据找到的存在目标的先验框的索引从模型预测结果中提取对应的分类信息
        classification_for_object = tf.gather_nd(classification, indices_for_object)
        # 计算二元交叉熵损失，用于存在目标的先验框
        # keras.backend.binary_crossentropy 函数用于计算真实标签和预测结果之间的二元交叉熵损失
        cls_loss_for_object = keras.backend.binary_crossentropy(labels_for_object, classification_for_object)
        # 找出实际上为背景的先验框的索引
        # 使用 tf.where 函数和 keras.backend.equal 检查哪些先验框的状态是0，即被标记为背景
        indices_for_back = tf.where(keras.backend.equal(anchor_state, 0))
        # 使用索引获取实际上为背景的先验框的真实标签
        # 使用 tf.gather_nd 函数根据背景先验框的索引从 labels 中提取对应的标签
        labels_for_back = tf.gather_nd(labels, indices_for_back)
        # 使用索引获取实际上为背景的先验框的模型预测结果
        # 使用 tf.gather_nd 函数根据背景先验框的索引从模型预测结果中提取相应数据
        classification_for_back = tf.gather_nd(classification, indices_for_back)
        # 计算二元交叉熵损失，用于实际上为背景的先验框
        # 使用 keras.backend.binary_crossentropy 函数计算背景先验框的真实标签和模型预测之间的二元交叉熵损失
        cls_loss_for_back = keras.backend.binary_crossentropy(labels_for_back, classification_for_back)
        # 计算正样本的数量，并将其转换为浮点数，最小值设置为1，防止分母为0
        # 使用 keras.backend.shape 获取存在目标的先验框（正样本）的数量
        # keras.backend.cast 将该数量转换为浮点数，以用于后续的损失计算
        # keras.backend.maximum 确保这个数量至少为1，避免在损失计算中出现除以0的情况
        normalizer_pos = keras.backend.maximum(keras.backend.cast(keras.backend.shape(indices_for_object)[0], keras.backend.floatx()), 1.0)
        # 计算负样本的数量，并将其转换为浮点数，最小值设置为1，防止分母为0
        # 使用 keras.backend.shape 获取背景先验框（负样本）的数量
        # keras.backend.cast 将该数量转换为浮点数，以用于后续的损失计算
        # keras.backend.maximum 确保这个数量至少为1，避免在损失计算中出现除以0的情况
        normalizer_neg = keras.backend.maximum(keras.backend.cast(keras.backend.shape(indices_for_back)[0], keras.backend.floatx()), 1.0)
        # 计算存在目标的先验框的分类损失，然后除以正样本数量
        # 使用 keras.backend.sum 对所有存在目标的先验框的损失进行求和
        # 然后除以 normalizer_pos（正样本数量），以平均这些损失
        cls_loss_for_object = keras.backend.sum(cls_loss_for_object) / normalizer_pos
        # 计算实际上为背景的先验框的分类损失，乘以权重比例后除以负样本数量
        # 使用 keras.backend.sum 对所有背景先验框的损失进行求和
        # 然后乘以正负样本比例（ratio）并除以 normalizer_neg（负样本数量）
        # 这样做是为了在损失函数中平衡正负样本的贡献
        cls_loss_for_back = ratio * keras.backend.sum(cls_loss_for_back) / normalizer_neg
        # 计算并返回总的分类损失，包括存在目标和实际上为背景的损失
        # 将存在目标的先验框损失和背景先验框损失相加，得到总的分类损失
        loss = cls_loss_for_object + cls_loss_for_back
        # 返回总的分类损失
        return loss

    # 返回内部定义的具体分类损失计算函数
    return _cls_loss



# 定义 smooth L1 损失函数，带有可调整的平滑参数 sigma
def smooth_l1(sigma=1.0):
    # 计算平滑参数 sigma 的平方值
    sigma_squared = sigma ** 2

    # 定义具体的 smooth L1 损失计算函数 _smooth_l1，接受真实标签 y_true 和预测结果 y_pred
    def _smooth_l1(y_true, y_pred):
        # 获取模型的预测结果，真实标签的回归目标，以及先验框的状态信息
        # 模型的预测结果，格式为 [batch_size, num_anchor, 4]
        regression = y_pred
        # 真实标签的回归目标，格式为 [batch_size, num_anchor, 4+1]，其中最后一维是先验框的状态信息
        regression_target = y_true[:, :, :-1]
        # 先验框的状态信息，-1 表示需要忽略的先验框，0 表示背景，1 表示存在目标
        anchor_state = y_true[:, :, -1]

        # 找到正样本的索引
        # 使用 TensorFlow 的 where 函数找到满足条件 anchor_state == 1 的索引
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        # 使用索引获取正样本的预测结果 regression 和真实回归目标 regression_target
        # 从整个 batch 的预测结果中筛选出正样本的预测结果
        regression = tf.gather_nd(regression, indices)
        # 从整个 batch 的真实回归目标中筛选出正样本的真实回归目标
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # 计算预测结果和真实回归目标的差异
        # regression_diff 存储了每个正样本的预测值与真实回归目标的差异
        regression_diff = regression - regression_target
        # 取 regression_diff 的绝对值，以确保差异值都为正
        regression_diff = keras.backend.abs(regression_diff)

        # 使用条件判断计算 smooth L1 loss
        # 使用 TensorFlow 的 where 函数，根据条件判断计算 smooth L1 loss
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # 计算损失值并进行归一化
        # 计算正样本的数量作为归一化因子
        # 使用 TensorFlow 的 maximum 函数取正样本数量和1的较大值
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        # 将正样本数量转换为浮点型
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        # 计算 smooth L1 loss 的总和，并除以归一化因子
        loss = keras.backend.sum(regression_loss) / normalizer

        # 返回 smooth L1 损失值
        return loss

    # 返回具体的 smooth L1 损失计算函数
    return _smooth_l1


# 定义分类与回归损失函数，其中 num_classes 为类别数
def class_loss_regr(num_classes):
    # 定义小常数 epsilon 用于避免分母为零的情况
    epsilon = 1e-4

    # 定义具体的分类与回归损失计算函数 class_loss_regr_fixed_num，接受真实标签 y_true 和预测结果 y_pred
    def class_loss_regr_fixed_num(y_true, y_pred):
        # 计算分类与回归损失

        # x 为真实标签中的回归目标与预测结果的差异
        x = y_true[:, :, 4 * num_classes:] - y_pred

        # x_abs 为 x 的绝对值，x_bool 为 x_abs 是否小于等于 1.0 的布尔值
        # 计算 x 的绝对值
        x_abs = K.abs(x)
        # 使用 TensorFlow 的 cast 函数将 x_abs 是否小于等于 1.0 的布尔值转换为 float32 类型
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')

        # 计算最终的分类与回归损失
        # y_true[:, :, :4 * num_classes] 为真实标签中的分类与回归目标
        # x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5) 为分类与回归损失的组合计算
        # 4 * K.sum(...) 为分类与回归损失的总和，其中权重系数为 4
        # K.sum(epsilon + y_true[:, :, :4 * num_classes]) 为分类标签的总和与小常数 epsilon 之和
        # 最终得到的 loss 进行了归一化操作
        loss = 4 * K.sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :4 * num_classes])

        # 返回最终的分类与回归损失值
        return loss

    # 返回具体的分类与回归损失计算函数 class_loss_regr_fixed_num
    return class_loss_regr_fixed_num


# 定义分类损失函数，接受真实标签 y_true 和预测结果 y_pred
def class_loss_cls(y_true, y_pred):
    # 使用 Keras 的 categorical_crossentropy 函数计算分类损失
    return K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))


# 定义函数获取调整后的图像大小
# 接受原始图像的宽度、高度以及最小边的大小，默认为 600
def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        # 如果原始图像的宽度小于等于高度，则按照比例调整高度和宽度
        # 计算宽度和高度的调整比例
        f = float(img_min_side) / width
        # 根据比例调整高度和宽度
        resized_height = int(f * height)
        resized_width = int(img_min_side)

    else:
        # 如果原始图像的宽度大于高度，则按照比例调整宽度和高度
        # 计算宽度和高度的调整比例
        f = float(img_min_side) / height
        # 根据比例调整宽度和高度
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    # 返回调整后的宽度和高度
    return resized_width, resized_height


# 定义函数获取卷积神经网络的输出长度

# 接受原始图像的宽度和高度
def get_img_output_length(width, height):
    # 定义内部函数 get_output_length，接受输入长度 input_length
    def get_output_length(input_length):
        # 设置卷积核的大小、填充大小、步幅大小
        filter_sizes = [7, 3, 1, 1]
        padding = [3, 1, 0, 0]
        stride = 2

        # 循环计算每一层的输出长度
        for i in range(4):
            # 使用公式计算卷积神经网络的输出长度
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1

        # 返回计算得到的输出长度
        return input_length

    # 分别计算宽度和高度的输出长度并返回
    return get_output_length(width), get_output_length(height)


# 定义名为 Generator 的类
class Generator(object):

    # 初始化方法，接收 bbox_util、train_lines、num_classes、solid 和 solid_shape 等参数
    def __init__(self, bbox_util, train_lines, num_classes, solid, solid_shape=[600, 600]):
        # 将传入的参数保存为对象属性
        self.bbox_util = bbox_util
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.num_classes = num_classes
        self.solid = solid
        self.solid_shape = solid_shape

    # 实时数据增强的随机预处理方法
    def get_random_data(self, annotation_line, random=True, jitter=.1, hue=.1, sat=1.1, val=1.1, proc_img=True):
        '''r实时数据增强的随机预处理'''
        # 将标注信息按空格分隔
        line = annotation_line.split()
        # 打开图像文件
        image = Image.open(line[0])
        # 获取图像的宽度和高度
        iw, ih = image.size

        # 根据是否使用指定形状，确定目标宽度和高度
        # 打开图像文件
        image = Image.open(line[0])
        # 获取图像的宽度和高度
        iw, ih = image.size

        # 根据是否使用指定形状，确定目标宽度和高度
        if self.solid:
            w, h = self.solid_shape
        else:
            # 获取新的图像大小
            w, h = get_new_img_size(iw, ih)

        # 从标注信息中提取边界框的坐标信息
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 调整图像大小
        # 计算新的宽高比例，并进行随机缩放和调整
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        # 随机缩放因子
        scale = rand(.9, 1.1)
        # 根据宽高比例调整图像大小
        if new_ar < 1:
            # 当新宽高比例小于1时，以高度为基准进行缩放
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            # 当新宽高比例大于等于1时，以宽度为基准进行缩放
            nw = int(scale * w)
            nh = int(nw / new_ar)
        # 使用双三次插值方法进行图像缩放
        image = image.resize((nw, nh), Image.BICUBIC)

        # 随机确定图像在新画布上的位置
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        # 创建一个新的画布，填充为灰色(128, 128, 128)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        # 将调整大小的图像粘贴到新的画布上
        new_image.paste(image, (dx, dy))
        # 更新图像为调整后的图像
        image = new_image

        # 以50%概率水平翻转图像
        # 以50%概率水平翻转图像
        flip = rand() < .5
        if flip:
            # 如果 flip 为 True，使用 PIL 库中的水平翻转操作
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 对图像进行颜色扭曲，包括色调变化、饱和度变化和明度变化
        # 随机调整图像的色调、饱和度和明度
        # 随机生成色调的调整值
        hue = rand(-hue, hue)
        # 随机生成饱和度的调整值
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        # 随机生成明度的调整值
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)

        # 将图像从 RGB 色彩空间转换为 HSV 色彩空间
        x = rgb_to_hsv(np.array(image) / 255.)

        # 调整图像的色调
        # 加上随机生成的色调调整值
        x[..., 0] += hue
        # 处理超出范围的值，减去1使其在[0, 1]范围内
        x[..., 0][x[..., 0] > 1] -= 1
        # 处理超出范围的值，加上1使其在[0, 1]范围内
        x[..., 0][x[..., 0] < 0] += 1

        # 调整图像的饱和度和明度
        # 将图像的饱和度乘以随机生成的饱和度调整值
        x[..., 1] *= sat
        # 将图像的明度乘以随机生成的明度调整值
        x[..., 2] *= val

        # 将调整后的图像转换回 RGB 色彩空间，并将数值范围限制在 [0, 1] 之间
        # 将调整后的图像的像素值限制在 [0, 1] 范围内
        # 大于1的值设为1
        x[x > 1] = 1
        # 小于0的值设为0
        x[x < 0] = 0
        # 将调整后的图像从 HSV 色彩空间转回 RGB 色彩空间，转换为 numpy 数组，范围为 0 到 255
        image_data = hsv_to_rgb(x) * 255

        # 对边界框进行修正，确保在图像范围内
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            # 随机打乱边界框的顺序
            np.random.shuffle(box)
            # 调整边界框的坐标，考虑图像的缩放和平移
            # 选取所有边界框的左上角和右下角 x 坐标，然后乘以图像宽度的缩放比例，最后加上随机水平平移量
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            # 选取所有边界框的左上角和右下角 y 坐标，然后乘以图像高度的缩放比例，最后加上随机垂直平移量
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

            # 如果图像被翻转，调整边界框的坐标
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            # 限制边界框的坐标在图像范围内
            # 将左上角 x 坐标小于0的值设置为0，确保边界框不超出图像左边界
            box[:, 0:2][box[:, 0:2] < 0] = 0
            # 将右下角 x 坐标大于图像宽度的值设置为图像宽度，确保边界框不超出图像右边界
            box[:, 2][box[:, 2] > w] = w
            # 将右下角 y 坐标大于图像高度的值设置为图像高度，确保边界框不超出图像下边界
            box[:, 3][box[:, 3] > h] = h
            # 计算边界框的宽度和高度，并去除无效的边界框
            # 计算每个边界框的宽度
            box_w = box[:, 2] - box[:, 0]
            # 计算每个边界框的高度
            box_h = box[:, 3] - box[:, 1]
            # 使用逻辑与操作，保留宽度和高度均大于1的边界框，去除无效的边界框
            box = box[np.logical_and(box_w > 1, box_h > 1)]

            # 将调整后的边界框信息保存到 box_data
            box_data[:len(box)] = box
        # 如果图像中没有有效的边界框，则返回空列表
        if len(box) == 0:
            return image_data, []

        # 如果 box_data 的前四列中有大于零的值，表示边界框有效
        if (box_data[:, :4] > 0).any():
            # 返回处理后的图像数据和有效的边界框信息
            return image_data, box_data
        else:
            # 如果图像中没有有效的边界框，返回处理后的图像数据和空列表
            return image_data, []

    def generate(self):
        while True:
            # 打乱训练数据集
            shuffle(self.train_lines)
            # 复制打乱后的训练数据集
            lines = self.train_lines

            # 遍历每个标注信息
            for annotation_line in lines:
                # 获取随机处理后的图像和边界框信息
                img, y = self.get_random_data(annotation_line)
                # 获取图像的高度、宽度和通道数
                height, width, _ = np.shape(img)

                # 如果边界框信息为空，则继续下一次循环
                if len(y) == 0:
                    continue

                # 对边界框坐标进行归一化
                # 将边界框坐标转换为浮点型，并进行归一化
                boxes = np.array(y[:, :4], dtype=np.float32)
                # 将边界框左上角 x 坐标归一化
                boxes[:, 0] = boxes[:, 0] / width
                # 将边界框左上角 y 坐标归一化
                boxes[:, 1] = boxes[:, 1] / height
                # 将边界框右下角 x 坐标归一化
                boxes[:, 2] = boxes[:, 2] / width
                # 将边界框右下角 y 坐标归一化
                boxes[:, 3] = boxes[:, 3] / height

                # 检查归一化后的边界框的宽度和高度是否合法
                # 计算归一化后的边界框的高度和宽度
                box_heights = boxes[:, 3] - boxes[:, 1]
                box_widths = boxes[:, 2] - boxes[:, 0]

                # 检查边界框的高度和宽度是否合法，如果不合法则跳过当前循环
                if (box_heights <= 0).any() or (box_widths <= 0).any():
                    continue

                # 更新边界框信息
                y[:, :4] = boxes[:, :4]

                # 获取先验框
                anchors = get_anchors(get_img_output_length(width, height), width, height)

                # 计算真实框对应的先验框，以及这个先验框应当有的预测结果
                assignment = self.bbox_util.assign_boxes(y, anchors)

                # 设置每个图像生成的区域数量
                num_regions = 256

                # 获取分类标签和回归目标，这些信息是从边界框分配得到的
                classification = assignment[:, 4]
                regression = assignment[:, :]

                # 处理正样本数量过多的情况
                # 根据分类标签找到正样本，如果正样本数量超过设定的阈值，则随机选择一部分样本标记为无效
                mask_pos = classification[:] > 0
                # 计算具有正分类标签的样本数量
                num_pos = len(classification[mask_pos])

                # 如果正样本数量超过设定的阈值，则随机选择一部分正样本标记为无效
                if num_pos > num_regions / 2:
                    # 从具有正分类标签的样本中随机选择一部分样本的索引，使得保留的正样本数量不超过 `num_regions / 2`
                    val_locs = random.sample(range(num_pos), int(num_pos - num_regions / 2))
                    # 将被选择的正样本的分类标签设为 -1，表示这些样本在训练中不会被用作正样本
                    classification[mask_pos][val_locs] = -1
                    # 将被选择的正样本的回归目标的最后一个元素设为 -1，表示这些样本在回归目标中不会被用作正样本
                    regression[mask_pos][val_locs, -1] = -1

                # 处理负样本数量过多的情况
                # 根据分类标签找到负样本
                mask_neg = classification[:] == 0
                # 计算具有负分类标签的样本数量
                num_neg = len(classification[mask_neg])

                # 如果负样本数量和正样本数量之和超过设定的总区域数量，则随机选择一部分负样本标记为无效
                if len(classification[mask_neg]) + num_pos > num_regions:
                    # 从具有负分类标签的样本中随机选择一部分样本的索引，使得保留的样本数量不超过设定的总区域数量
                    val_locs = random.sample(range(num_neg), int(num_neg - num_pos))
                    # 将被选择的负样本的分类标签设为 -1，表示这些样本在训练中不会被用作负样本
                    classification[mask_neg][val_locs] = -1

                # 将分类标签和回归目标进行形状调整
                # 调整分类标签的形状，使之适应模型的输入
                classification = np.reshape(classification, [-1, 1])
                # 调整回归目标的形状，使之适应模型的输入
                regression = np.reshape(regression, [-1, 5])

                # 构建输入数据和目标数据
                # 将图像数据转换为 NumPy 数组
                tmp_inp = np.array(img)
                # 将分类标签和回归目标转换为 NumPy 数组，并在第0维度添加一个维度
                tmp_targets = [
                    np.expand_dims(np.array(classification, dtype=np.float32), 0),
                    np.expand_dims(np.array(regression, dtype=np.float32), 0)
                ]

                # 产生生成器的输出
                yield preprocess_input(np.expand_dims(tmp_inp, 0)), tmp_targets, np.expand_dims(y, 0)
