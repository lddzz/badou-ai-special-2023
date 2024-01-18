# 导入print函数的新特性,使代码在Python 2.x和Python 3.x中都能运行
from __future__ import print_function
# 导入绝对导入的新特性
from __future__ import absolute_import
# 导入警告处理模块
import warnings
# 导入处理数组和矩阵的NumPy库
import numpy as np
# 导入Keras模型和层的相关模块
from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
# 导入获取输入源的函数、转换模型中所有核心的函数和获取文件的函数
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
# 导入Keras的后端模块,并用K表示
from keras import backend as K
# 导入用于解码模型预测结果的函数
from keras.applications.imagenet_utils import decode_predictions
# 导入图像预处理模块
from keras.preprocessing import image


# 定义一个函数conv2d_bn,用于创建带有卷积、批量归一化和激活函数的层
# x: 输入张量,表示要在其上应用卷积、批量归一化和激活函数的张量。
# filters: 表示卷积层中的滤波器数量。
# num_row 和 num_col: 表示卷积核的行数和列数。
# strides: 表示卷积的步幅,默认为(1, 1)。
# padding: 表示填充方式,默认为'same'。
# name: 表示层的名称,如果提供了名称,将用于构造批量归一化和卷积层的名称。
def conv2d_bn(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    # 如果提供了名称,构造批量归一化和卷积层的名称,否则将它们设为 None
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    # 使用 Keras 中的 Conv2D 函数创建二维卷积层,name=conv_name
    # 设置滤波器数量filters、卷积核kernel_size大小num_row, num_col、步幅strides、填充方式padding等参数
    # 并将卷积层应用到输入张量 x 上
    x = Conv2D(filters=filters,
               kernel_size=(num_row, num_col),
               strides=strides,
               padding=padding,
               use_bias=False,
               name=conv_name)(x)
    # 使用 Keras 中的 BatchNormalization 函数创建批量归一化层
    # 设置 scale 为 False 表示不使用缩放,并将该层应用到之前的输出 x 上
    x = BatchNormalization(scale=False, name=bn_name)(x)
    # 使用 Keras 中的 Activation 函数,将激活函数设置为 ReLU
    # 并将该层应用到之前的输出 x 上
    x = Activation('relu', name=bn_name)(x)
    # 返回处理后的张量 x,包含了卷积、批量归一化和激活函数的效果
    return x


# 定义 InceptionV3 模型函数,输入input_shape图片尺寸299, 299, 3;分类为1000
def InceptionV3(input_shape=[299, 299, 3], classes=1000):
    # 输入层
    img_input = Input(input_shape)

    # 第一个模块
    # 使用 conv2d_bn 函数创建一个卷积层,32个过滤器,过滤器大小为3x3,步幅为2x2,padding为'valid'
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    # 使用 conv2d_bn 函数创建第二个卷积层,32个过滤器,过滤器大小为3x3,padding为'valid'
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    # 使用 conv2d_bn 函数创建第三个卷积层,64个过滤器,过滤器大小为3x3
    x = conv2d_bn(x, 64, 3, 3)
    # 使用 MaxPooling2D 函数创建一个最大池化层,池化窗口大小为3x3,步幅为2x2
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 第二个模块
    # 使用 conv2d_bn 函数创建第一个卷积层,80个过滤器,过滤器大小为1x1,padding为'valid'
    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    # 使用 conv2d_bn 函数创建第二个卷积层,192个过滤器,过滤器大小为3x3,padding为'valid'
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    # 使用 MaxPooling2D 函数创建一个最大池化层,池化窗口大小为3x3,步幅为2x2
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    # 第一个 Inception 模块
    # 包含三个部分(part1, part2, part3)

    # part1
    # 第1分叉:1x1
    # 使用 conv2d_bn 函数创建卷积层,64个过滤器,过滤器大小为1x1
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    # 第2分叉:1x1->5x5(3x3)
    # 使用 conv2d_bn 函数创建第一个卷积层,48个过滤器,过滤器大小为1x1
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层,64个过滤器,过滤器大小为5x5
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    # 第3分叉:1x1->3x3->3x3
    # 使用 conv2d_bn 函数创建第一个卷积层,64个过滤器,过滤器大小为1x1
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层,96个过滤器,过滤器大小为3x3
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    # 使用 conv2d_bn 函数创建第三个卷积层,96个过滤器,过滤器大小为3x3
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)

    # 第4分叉:pooling->1x1
    # 使用 AveragePooling2D 函数创建一个平均池化层,池化窗口大小为3x3,步幅为1x1,padding为'same'
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 使用 conv2d_bn 函数创建一个卷积层,32个过滤器,过滤器大小为1x1
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    # 使用 layers.concatenate 函数连接所有分支的输出,axis=3 表示在通道维度上连接,命名为mixed0
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3, name='mixed0')

    # part2
    # 第1分叉:1x1
    # 使用 conv2d_bn 函数创建一个卷积层,64个过滤器,过滤器大小为1x1
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    # 第2分叉:1x1->5x5(3x3)
    # 使用 conv2d_bn 函数创建第一个卷积层,48个过滤器,过滤器大小为1x1
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层,64个过滤器,过滤器大小为5x5
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    # 第3分叉:1x1->3x3->3x3
    # 使用 conv2d_bn 函数创建第一个卷积层,64个过滤器,过滤器大小为1x1
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层,96个过滤器,过滤器大小为3x3
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    # 使用 conv2d_bn 函数创建第三个卷积层,96个过滤器,过滤器大小为3x3
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)

    # 第4分叉:pooling->1x1
    # 使用 AveragePooling2D 函数创建一个平均池化层,池化窗口大小为3x3,步幅为1x1,padding为'same'
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 使用 conv2d_bn 函数创建一个卷积层,64个过滤器,过滤器大小为1x1
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    # 使用 layers.concatenate 函数连接所有分支的输出,axis=3 表示在通道维度上连接,命名为mixed1
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3, name='mixed1')

    # part3
    # 第1分叉:1x1
    # 使用 conv2d_bn 函数创建一个卷积层,64个过滤器,过滤器大小为1x1
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    # 第2分叉:1x1->5x5(3x3)
    # 使用 conv2d_bn 函数创建第一个卷积层,48个过滤器,过滤器大小为1x1
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层,64个过滤器,过滤器大小为5x5
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    # 第3分叉:1x1->3x3->3x3
    # 使用 conv2d_bn 函数创建第一个卷积层,64个过滤器,过滤器大小为1x1
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层,96个过滤器,过滤器大小为3x3
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    # 使用 conv2d_bn 函数创建第三个卷积层,96个过滤器,过滤器大小为3x3
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)

    # 第4分叉:pooling->1x1
    # 使用 AveragePooling2D 函数创建一个平均池化层,池化窗口大小为3x3,步幅为1x1,padding为'same'
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 使用 conv2d_bn 函数创建一个卷积层,64个过滤器,过滤器大小为1x1
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    # 使用 layers.concatenate 函数连接所有分支的输出,axis=3 表示在通道维度上连接,命名为mixed2
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3, name='mixed2')

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    # 第二个 Inception 模块
    # 包含五个部分(part1, part2, part3,part4, part5)
    # part1

    # 第1分叉:3x3
    # 使用 conv2d_bn 函数创建一个卷积层,384个过滤器,过滤器大小为3x3,步幅为2x2,padding为'valid'
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    # 第2分叉:1x1->3x3->3x3
    # 使用 conv2d_bn 函数创建第一个卷积层,64个过滤器,过滤器大小为1x1
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层,96个过滤器,过滤器大小为3x3
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    # 使用 conv2d_bn 函数创建第三个卷积层,96个过滤器,过滤器大小为3x3,步幅为2x2,padding为'valid'
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    # 第3分叉:pooling
    # 使用 MaxPooling2D 函数创建一个最大池化层,池化窗口大小为3x3,步幅为2x2
    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 使用 layers.concatenate 函数连接所有分支的输出,axis=3 表示在通道维度上连接,命名为mixed3
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # part2
    # 第1分叉:1x1
    # 使用 conv2d_bn 函数创建一个卷积层,192个过滤器,过滤器大小为1x1
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 第2分叉:1x1->1x7->7x1
    # 使用 conv2d_bn 函数创建第一个卷积层,128个过滤器,过滤器大小为1x1
    branch7x7 = conv2d_bn(x, 128, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层,128个过滤器,过滤器大小为1x7
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    # 使用 conv2d_bn 函数创建第三个卷积层,192个过滤器,过滤器大小为7x1
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 第3分叉:1x1->7x1->1x7->7x1->1x7
    # 使用 conv2d_bn 函数创建第一个卷积层,128个过滤器,过滤器大小为1x1
    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层,128个过滤器,过滤器大小为7x1
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    # 使用 conv2d_bn 函数创建第三个卷积层,128个过滤器,过滤器大小为1x7
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    # 使用 conv2d_bn 函数创建第四个卷积层,128个过滤器,过滤器大小为7x1
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    # 使用 conv2d_bn 函数创建第五个卷积层,192个过滤器,过滤器大小为1x7
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    # 第4分叉:pooling->1x1
    # 使用 AveragePooling2D 函数创建一个平均池化层,池化窗口大小为3x3,步幅为1x1,padding为'same'
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 使用 conv2d_bn 函数创建一个卷积层,192个过滤器,过滤器大小为1x1
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    # 使用 layers.concatenate 函数连接所有分支的输出,axis=3 表示在通道维度上连接,命名为mixed4
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed4')

    # part3 and part4
    # 循环两次
    for i in range(2):
        # 第1分叉:1x1
        # 使用 conv2d_bn 函数创建一个卷积层,192个过滤器,过滤器大小为1x1
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        # 第2分叉:1x1->1x7->7x1
        # 使用 conv2d_bn 函数创建第一个卷积层,160个过滤器,过滤器大小为1x1
        branch7x7 = conv2d_bn(x, 160, 1, 1)
        # 使用 conv2d_bn 函数创建第二个卷积层,160个过滤器,过滤器大小为1x7
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        # 使用 conv2d_bn 函数创建第三个卷积层,192个过滤器,过滤器大小为7x1
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        # 第3分叉:1x1->7x1->1x7->7x1->1x7
        # 使用 conv2d_bn 函数创建第一个卷积层,160个过滤器,过滤器大小为1x1
        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        # 使用 conv2d_bn 函数创建第二个卷积层,160个过滤器,过滤器大小为7x1
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        # 使用 conv2d_bn 函数创建第三个卷积层,160个过滤器,过滤器大小为1x7
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        # 使用 conv2d_bn 函数创建第四个卷积层,160个过滤器,过滤器大小为7x1
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        # 使用 conv2d_bn 函数创建第五个卷积层,192个过滤器,过滤器大小为1x7
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        # 第4分叉:pooling->1x1
        # 使用 AveragePooling2D 函数创建一个平均池化层,池化窗口大小为3x3,步幅为1x1,padding为'same'
        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        # 使用 conv2d_bn 函数创建一个卷积层,192个过滤器,过滤器大小为1x1
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        # 使用 layers.concatenate 函数连接所有分支的输出,axis=3 表示在通道维度上连接,命名为'mixed' + str(5 + i)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed' + str(5 + i))

    # part5
    # 第1分叉:1x1
    # 使用 conv2d_bn 函数创建一个卷积层，192个过滤器，过滤器大小为1x1
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 第2分叉:1x1->1x7->7x1
    # 使用 conv2d_bn 函数创建第一个卷积层，192个过滤器，过滤器大小为1x1
    branch7x7 = conv2d_bn(x, 192, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层，192个过滤器，过滤器大小为1x7
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    # 使用 conv2d_bn 函数创建第三个卷积层，192个过滤器，过滤器大小为7x1
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 第3分叉:1x1->7x1->1x7->7x1->1x7
    # 使用 conv2d_bn 函数创建第一个卷积层，192个过滤器，过滤器大小为1x1
    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层，192个过滤器，过滤器大小为7x1
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    # 使用 conv2d_bn 函数创建第三个卷积层，192个过滤器，过滤器大小为1x7
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    # 使用 conv2d_bn 函数创建第四个卷积层，192个过滤器，过滤器大小为7x1
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    # 使用 conv2d_bn 函数创建第五个卷积层，192个过滤器，过滤器大小为1x7
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    # 第4分叉:pooling->1x1
    # 使用 AveragePooling2D 函数创建一个平均池化层，池化窗口大小为3x3，步幅为1x1,padding='same'
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 使用 conv2d_bn 函数创建一个卷积层，192个过滤器，过滤器大小为1x1
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    # 使用 layers.concatenate 函数连接所有分支的输出，axis=3 表示在通道维度上连接,命名为mixed7
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed7')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    # 第三个 Inception 模块
    # 包含两个部分(part1, part2)

    # part1
    # 第1分叉:1x1->3x3
    # 使用 conv2d_bn 函数创建一个卷积层，192个过滤器，过滤器大小为1x1
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层，320个过滤器，过滤器大小为3x3，步幅为2x2，padding='valid' 表示不使用填充
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    # 第2分叉:1x1->1x7->7x1->3x3
    # 使用 conv2d_bn 函数创建一个卷积层，192个过滤器，过滤器大小为1x1
    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    # 使用 conv2d_bn 函数创建第二个卷积层，192个过滤器，过滤器大小为1x7
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    # 使用 conv2d_bn 函数创建第三个卷积层，192个过滤器，过滤器大小为7x1
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    # 使用 conv2d_bn 函数创建第四个卷积层，192个过滤器，过滤器大小为3x3，步幅为2x2，padding='valid' 表示不使用填充
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    # 第3分叉:pooling
    # 使用 MaxPooling2D 函数创建一个最大池化层，池化窗口大小为3x3，步幅为2x2
    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 使用 layers.concatenate 函数连接所有分支的输出，axis=3 表示在通道维度上连接,命名为mixed8
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # part2 and part3
    # 使用 for 循环两次，构建两个分支
    for i in range(2):
        # 第1分叉:1x1
        # 使用 conv2d_bn 函数创建一个卷积层，320个过滤器，过滤器大小为1x1
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        # 第2分叉:1x1->1x3连接3x1
        # 使用 conv2d_bn 函数创建一个卷积层，384个过滤器，过滤器大小为1x1
        branch3x3 = conv2d_bn(x, 384, 1, 1)
        # 使用 conv2d_bn 函数创建第二个卷积层，384个过滤器，过滤器大小为1x3
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        # 使用 conv2d_bn 函数创建第三个卷积层，384个过滤器，过滤器大小为3x1
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        # 使用 layers.concatenate 函数在通道维度上连接两个分支,axis=3 表示在通道维度上连接,命名为'mixed9_'+str(i)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        # 第3分叉:1x1->3x3->1x3连接3x1
        # 使用 conv2d_bn 函数创建一个卷积层，448个过滤器，过滤器大小为1x1
        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        # 使用 conv2d_bn 函数创建一个卷积层，384个过滤器，过滤器大小为3x3
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        # 使用 conv2d_bn 函数创建第二个卷积层，384个过滤器，过滤器大小为1x3
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        # 使用 conv2d_bn 函数创建第三个卷积层，384个过滤器，过滤器大小为3x1
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        # 使用 layers.concatenate 函数在通道维度上连接两个分支,axis=3 表示在通道维度上连接
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        # 第4分叉:pooling->1x1
        # 使用 AveragePooling2D 函数创建一个平均池化层，池化窗口大小为3x3，步幅为1x1,padding='same'
        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        # 使用 conv2d_bn 函数创建一个卷积层，192个过滤器，过滤器大小为1x1
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        # 使用 layers.concatenate 函数连接所有分支的输出，axis=3 表示在通道维度上连接,命名为'mixed' + str(9 + i)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed' + str(9 + i))

    # 最后的全连接层
    # 使用 GlobalAveragePooling2D 函数创建一个全局平均池化层,name='avg_pool'
    x = GlobalAveragePooling2D(name='avgpool')(x)
    # 使用 Dense 函数创建一个全连接层，输出类别数为 classes，激活函数为 softmax,name='predictions'
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # 输入和输出构成模型
    # 将输入和输出构建为一个模型,name='inception_v3'
    inputs = img_input
    model = Model(inputs=inputs, outputs=x, name='inception_v3')

    # 返回模型
    return model


# 定义图像预处理函数
def preprocess_input(x):
    # 像素值缩放到 [0, 1] 范围
    x /= 255.
    # 中心化到 [-0.5, 0.5] 范围
    x -= 0.5
    # 标准化到 [-1, 1] 范围
    x *= 2
    return x


if __name__ == '__main__':
    # 创建 InceptionV3 模型
    model = InceptionV3()

    # 加载预训练权重
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    # 加载图像并进行预处理
    # 图像路径
    img_path = 'c.jpg'
    # 加载图像并调整大小为模型所需的输入尺寸
    img = image.load_img(img_path, target_size=(299, 299))
    # 将图像转换为 NumPy 数组
    x = image.img_to_array(img)
    # 在第一个维度上添加一个维度，以符合模型输入的形状
    x = np.expand_dims(x, axis=0)
    # 对图像进行预处理
    x = preprocess_input(x)

    # 模型推理
    preds = model.predict(x)
    # 输出预测结果
    print(f"预测结果:{decode_predictions(preds, 1)}")
