# 导入 NumPy 库并使用别名 np
import numpy as np
# 导入 TensorFlow 库并使用别名 tf
import tensorflow as tf
# 从 PIL 库中导入 Image 模块
from PIL import Image


# 定义加载神经网络权重的函数，接受变量列表和权重文件路径两个参数
def load_weights(var_list, weights_file):
    # 使用二进制模式打开权重文件，并将文件对象赋给 fp
    with open(weights_file, "rb") as fp:
        # 从文件中读取 5 个 32 位整数，但忽略其值
        _ = np.fromfile(fp, dtype=np.int32, count=5)
        # 从文件中读取浮点数权重，并将其存储在 weights 变量中
        weights = np.fromfile(fp, dtype=np.float32)
    # 初始化指针位置为零
    ptr = 0
    # 初始化索引位置为零
    i = 0
    # 初始化一个空列表，用于存储赋值操作
    assign_ops = []
    # 进入一个循环，遍历变量列表，直到列表的倒数第二个元素
    while i < len(var_list) - 1:
        # 获取当前变量
        var1 = var_list[i]
        # 获取下一个变量
        var2 = var_list[i + 1]
        # 检查当前变量是否是卷积层
        if 'conv2d' in var1.name.split('/')[-2]:
            # 如果下一个变量是批归一化层，执行以下操作
            if 'batch_normalization' in var2.name.split('/')[-2]:
                # 从变量列表中提取批归一化参数
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                # 将批归一化参数存储在列表中
                batch_norm_vars = [beta, gamma, mean, var]
                # 遍历批归一化参数列表
                for var in batch_norm_vars:
                    # 获取当前参数的形状
                    shape = var.shape.as_list()
                    # 计算参数数量
                    num_params = np.prod(shape)
                    # 从权重数组中提取对应数量的参数，并根据形状重塑
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    # 更新指针位置
                    ptr += num_params
                    # 将赋值操作添加到列表中
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # 移动索引位置，因为加载了 4 个变量
                i += 4
            # 如果下一个变量是卷积层，执行以下操作
            elif 'conv2d' in var2.name.split('/')[-2]:
                # 获取偏置变量
                bias = var2
                # 获取偏置的形状信息
                bias_shape = bias.shape.as_list()
                # 计算偏置参数的总数
                bias_params = np.prod(bias_shape)
                # 从权重数组中提取对应数量的偏置参数，并根据形状重塑
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                # 更新指针位置
                ptr += bias_params
                # 将偏置的赋值操作添加到列表中
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # 加载了 1 个变量
                i += 1
            # 获取卷积层权重的形状信息
            shape = var1.shape.as_list()
            # 计算卷积层参数数量
            num_params = np.prod(shape)
            # 从权重数组中提取对应数量的权重，并根据形状重塑
            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # 转置权重，将其顺序调整为列主序
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            # 更新指针位置
            ptr += num_params
            # 将卷积层权重的赋值操作添加到列表中
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            # 增加索引，移至下一个变量
            i += 1
    # 返回赋值操作列表
    return assign_ops


# 定义一个函数letterbox_image，用于在保持长宽比的情况下将图像缩放到指定大小
# 接受image输入图像,size图像大小两个参数
def letterbox_image(image, size):
    # 获取输入图像的宽高
    image_w, image_h = image.size
    # 获取目标大小
    w, h = size
    # 计算按照长宽比进行缩放后的新宽度和新高度
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))
    # 使用双三次插值法进行图像缩放
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)
    # 创建新图像，填充灰色背景
    box_image = Image.new('RGB', size, (128, 128, 128))
    # 在新图像上粘贴缩放后的图像，使其居中
    box_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
    # 返回缩放后的图像
    return box_image


# 定义一个函数，用于在图像上绘制边界框
def draw_box(image, box):
    # 使用 TensorFlow 的 split 函数按照指定轴拆分 bbox
    xmin, ymin, xmax, ymax, label = tf.split(value=box, num_or_size_splits=5, axis=2)
    # 获取图像的高度和宽度
    height = tf.cast(tf.shape(image)[1], tf.float32)
    width = tf.cast(tf.shape(image)[2], tf.float32)
    # 计算归一化后的新坐标
    new_bbox = tf.concat(
        [tf.cast(ymin, tf.float32) / height, tf.cast(xmin, tf.float32) / width,
         tf.cast(ymax, tf.float32) / height, tf.cast(xmax, tf.float32) / width], 2)
    # 使用 TensorFlow 的 draw_bounding_boxes 函数在图像上绘制新的边界框
    new_image = tf.image.draw_bounding_boxes(image, new_bbox)
    # 使用 TensorFlow 的 tf.summary.image 函数将图像添加到 TensorBoard 可视化中
    tf.summary.image('input', new_image)


# 定义一个计算VOC平均精度（AP）的函数
def voc_ap(rec, prec):
    # 在列表开头插入 0.0
    rec.insert(0, 0.0)
    # 在列表末尾插入 1.0
    rec.append(1.0)
    # 复制列表
    mrec = rec[:]
    # 在列表开头插入 0.0
    prec.insert(0, 0.0)
    # 在列表末尾插入 0.0
    prec.append(0.0)
    # 复制列表
    mpre = prec[:]
    # 从倒数第二个元素开始逆向遍历，确保每个元素都不小于其后一个元素
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # 寻找不同的元素的索引
    i_list = [i for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1]]
    # 计算平均精度（AP）
    ap = 0.0
    for i in i_list:
        # 计算当前召回率区间的宽度乘以对应的精确率，然后累加得到总的AP
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    # 返回平均精度、修改后的召回率列表和修改后的精确率列表
    return ap, mrec, mpre
