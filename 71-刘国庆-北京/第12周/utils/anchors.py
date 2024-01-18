import keras  # 导入 Keras 库
import numpy as np  # 导入 NumPy 库，并命名为 np
from utils.config import Config  # 从 utils 模块的 config 文件中导入 Config 类

config = Config()  # 创建 Config 的实例


# 定义生成锚点的函数
def generate_anchors(sizes=None, ratios=None):
    # 如果 sizes 未指定，从配置中获取默认值
    if sizes is None:
        sizes = config.anchor_box_scales

    # 如果 ratios 未指定，从配置中获取默认值
    if ratios is None:
        ratios = config.anchor_box_ratios

    # 计算锚点的数量
    num_anchors = len(sizes) * len(ratios)
    # 初始化锚点数组
    anchors = np.zeros((num_anchors, 4))

    # 根据 sizes 更新锚点的宽度和高度
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T

    # 更新锚点数组以考虑不同的比例
    for i in range(len(ratios)):  # 遍历所有的宽高比
        # 调整锚点的宽度。对于每个比例，更新锚点矩阵的对应行的第三列(宽度)
        anchors[3 * i:3 * i + 3, 2] = anchors[3 * i:3 * i + 3, 2] * ratios[i][0]
        # 调整锚点的高度。对于每个比例，更新锚点矩阵的对应行的第四列(高度)
        anchors[3 * i:3 * i + 3, 3] = anchors[3 * i:3 * i + 3, 3] * ratios[i][1]

    # 调整锚点的 x 坐标
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    # 调整锚点的 y 坐标
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    # 返回生成的锚点
    return anchors


# 定义用于移动锚点的函数
def shift(shape, anchors, stride=config.rpn_stride):
    # 计算 x 方向的平移量
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    # 计算 y 方向的平移量
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
    # 生成网格
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # 将平移量重塑为一维数组
    # 将 shift_x 重塑成一维数组
    shift_x = np.reshape(shift_x, [-1])
    # 将 shift_y 重塑成一维数组
    shift_y = np.reshape(shift_y, [-1])

    # 组合平移量
    shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
    # 转置
    shifts = np.transpose(shifts)
    # 获取锚点的数量
    number_of_anchors = np.shape(anchors)[0]
    # 获取 shifts 数组第一维的大小
    k = np.shape(shifts)[0]

    # 应用平移量到锚点上
    # 将原始锚点和计算出的平移量组合，生成平移后的锚点
    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]),keras.backend.floatx())
    # 将平移后的锚点数组重塑为二维数组
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    # 返回移动后的锚点
    return shifted_anchors


# 定义获取锚点的函数
def get_anchors(shape, width, height):
    # 生成锚点
    anchors = generate_anchors()
    # 移动锚点
    network_anchors = shift(shape, anchors)
    # 将锚点坐标归一化到图像尺寸
    # 将锚点的 x 坐标归一化到图像宽度
    network_anchors[:, 0] = network_anchors[:, 0] / width
    # 将锚点的 y 坐标归一化到图像高度
    network_anchors[:, 1] = network_anchors[:, 1] / height
    # 将锚点的 x2 坐标(锚点的宽度)归一化到图像宽度
    network_anchors[:, 2] = network_anchors[:, 2] / width
    # 将锚点的 y2 坐标(锚点的高度)归一化到图像高度
    network_anchors[:, 3] = network_anchors[:, 3] / height
    # 将锚点坐标限制在0到1之间
    network_anchors = np.clip(network_anchors, 0, 1)
    # 返回处理后的锚点
    return network_anchors
