# 导入 NumPy 模块并使用别名 np
import numpy as np
# 导入 TensorFlow 模块并使用别名 tf
import tensorflow as tf
# 导入 os 模块
import os


# 定义名为 yolo 的类
class yolo:
    # 类的初始化函数，接收一些参数用于初始化类的属性
    # norm_epsilon: 方差加上极小的数，防止除以0的情况
    # norm_decay: 在预测时计算movingaverage时的衰减率
    # anchors_path: yoloanchor文件路径
    # classes_path: 数据集类别对应文件
    # pre_train: 是否使用预训练darknet53模型
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        # 将参数赋值给类的属性
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        # 调用get_anchors方法获取 anchors
        self.anchors = self.get_anchors()
        # 调用get_class方法获取类别信息
        self.classes = self.get_class()

    # ---------------------------------------#
    #   获取种类和先验框
    # ---------------------------------------#
    # 获取类别名字
    # 返回class_names: coco 数据集类别对应的名字
    def get_class(self):
        # 扩展用户目录，并打开类别文件
        classes_path = os.path.expanduser(self.classes_path)
        # 使用 with 语句打开文件，创建文件对象 f
        with open(classes_path) as f:
            # 逐行读取类别名字，去除首尾空白字符
            class_names = f.readlines()
        # 将读取的类别名字列表去除首尾空白字符后
        class_names = [c.strip() for c in class_names]
        # 返回类别名字列表
        return class_names

    # 获取 anchors
    def get_anchors(self):
        # 扩展用户目录，并打开 anchors 文件
        anchors_path = os.path.expanduser(self.anchors_path)
        # 使用 with 语句打开文件，创建文件对象 f
        with open(anchors_path) as f:
            # 读取一行 anchors 数据
            anchors = f.readline()
        # 将读取的 anchors 字符串转换为浮点数列表，并重新组织成 NumPy 数组
        anchors = [float(x) for x in anchors.split(',')]
        # 返回 NumPy 数组，通过 reshape 转换为二维数组
        return np.array(anchors).reshape(-1, 2)

    # ---------------------------------------#
    #   用于生成层
    # ---------------------------------------#
    # l2 正则化batch_normalization_layer
    # 对卷积层提取的 feature map 使用 batch normalization
    # 输入:
    # input_layer: 输入的四维 tensor,
    # name=None: batchnorm 层的名字
    # training=True: 是否为训练过程
    # norm_decay=0.99: 在预测时计算 moving average 时的衰减率
    # norm_epsilon=1e-3: 方差加上极小的数，防止除以 0 的情况
    # 输出:bn_layer: batch normalization 处理之后的 feature map
    def batch_normalization_layer(self, input_layer, name=None, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        # 使用 tf.layers.batch_normalization 对输入的 feature map 进行 batch normalization
        # inputs=input_layer:输入特征图
        # momentum=norm_decay:动量参数，用于计算移动平均
        # epsilon=norm_epsilon:一个小的常数，用于防止分母为零
        # center=True:是否进行中心化
        # scale=True:是否进行缩放
        # training=training:是否在训练模式下
        # name=name 批量归一化层的名称
        bn_layer = tf.layers.batch_normalization(
            inputs=input_layer,
            momentum=norm_decay,
            epsilon=norm_epsilon,
            center=True,
            scale=True,
            training=training,
            name=name
        )

        # 返回对 bn_layer处理后的结果,使用 leaky ReLU 激活函数,负数部分的斜率alpha为0.1
        return tf.nn.leaky_relu(bn_layer, alpha=0.1)

    # 这个就是用来进行卷积的
    # 这个函数用于进行卷积操作，采用 tf.layers.conv2d
    # 卷积后进行 batch normalization 和 leaky ReLU 激活函数
    # 根据卷积时的步长，处理降采样以及 padding 操作
    # 卷积核大小为 3，如果步长为 2，则相当于降采样
    # 对于大于 1 的步长，使用 padding 为 'VALID'，在四周进行一维 padding
    # inputs: 输入变量
    # filters_num: 卷积核数量
    # kernel_size: 卷积核大小
    # training: 是否为训练过程
    # use_bias=False: 是否使用偏置项
    # strides=1: 卷积核步长
    # name:名字
    # conv: 卷积之后的featuremap
    def conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        # 使用 tf.layers.conv2d 进行卷积操作，其中包括权重和偏置矩阵的初始化
        # 输入 inputs，卷积核数量为 filters_num，卷积核大小为 kernel_size
        # 步长为 [strides, strides]，采用 Glorot uniform 初始化权重
        # 根据步长选择 'SAME' 或 'VALID' 的 padding 方式
        # 使用 L2 正则化，是否使用偏置项由 use_bias 决定
        # 命名由name决定
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters_num,
            kernel_size=kernel_size,
            strides=[strides, strides],
            kernel_initializer=tf.glorot_uniform_initializer(),
            padding=('SAME' if strides == 1 else 'VALID'),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
            use_bias=use_bias,
            name=name)
        # 返回卷积后的 feature map
        return conv

    # 残差卷积Residual_block:3X3->batch normalization->1X1->batch normalization->3X3->batch normalization->
    # 残差卷积就是进行一次3X3的卷积，然后保存该卷积layer
    # 再进行一次1X1的卷积和一次3X3的卷积，并把这个结果加上layer作为最后的结果
    # 输入:
    # inputs: 输入变量,
    # filters_num: 卷积核数量
    # blocks_num: block的数量
    # conv_index: 为了方便加载预训练权重，统一命名序号
    # training=True: 是否为训练过程
    # norm_decay=0.99: 在预测时计算moving,average时的衰减率
    # norm_epsilon=1e-3: 方差加上极小的数，防止除以0的情况
    # 返回:
    # inputs: 经过残差网络处理后的结果
    def Residual_block(self, inputs, filters_num, blocks_num, conv_index, training=True, norm_decay=0.99,
                       norm_epsilon=1e-3):
        # 在输入feature map的长宽维度进行padding
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        # 调用conv2d_layer方法卷积
        # inputs: 输入变量
        # filters_num: 卷积核数量
        # kernel_size=3: 卷积核大小
        # use_bias=False: 是否使用偏置项
        # strides=2: 卷积核步长
        # name=f"conv2d_{str(conv_index)}":名字
        layer = self.conv2d_layer(
            inputs,
            filters_num,
            kernel_size=3,
            use_bias=False,
            strides=2,
            name=f"conv2d_{str(conv_index)}"
        )
        # 对卷积结果进行 batch normalization
        # layer: 输入的四维 tensor,
        # name="batch_normalization_" + str(conv_index): batchnorm 层的名字
        # training=training: 是否为训练过程
        # norm_decay=norm_decay: 在预测时计算 moving average 时的衰减率
        # norm_epsilon=norm_epsilon: 方差加上极小的数，防止除以 0 的情况
        layer = self.batch_normalization_layer(
            layer,
            name=f"batch_normalization_{str(conv_index)}",
            training=training,
            norm_decay=norm_decay,
            norm_epsilon=norm_epsilon
        )
        # 自增卷积索引
        conv_index += 1
        # 循环进行残差卷积操作
        for _ in range(blocks_num):
            shortcut = layer
            # 1x1 卷积，减少通道数
            # layer: 输入变量
            # filters_num // 2: 卷积核数量
            # kernel_size=1: 卷积核大小
            # strides=1: 卷积核步长
            # name="conv2d_" + str(conv_index):名字
            layer = self.conv2d_layer(
                layer,
                filters_num // 2,
                kernel_size=1,
                strides=1,
                name=f"conv2d_{str(conv_index)}"
            )
            # batch normalization
            # layer: 输入的四维 tensor,
            # name="batch_normalization_" + str(conv_index): batchnorm 层的名字
            # training=training: 是否为训练过程
            # norm_decay=norm_decay: 在预测时计算 moving average 时的衰减率
            # norm_epsilon=norm_epsilon: 方差加上极小的数，防止除以 0 的情况
            layer = self.batch_normalization_layer(
                layer,
                name=f"batch_normalization_{str(conv_index)}",
                training=training,
                norm_decay=norm_decay,
                norm_epsilon=norm_epsilon
            )
            # 自增卷积索引
            conv_index += 1
            # 3x3 卷积
            # layer: 输入变量
            # filters_num: 卷积核数量
            # kernel_size=3: 卷积核大小
            # strides=1: 卷积核步长
            # name="conv2d_" + str(conv_index):名字
            layer = self.conv2d_layer(
                layer,
                filters_num,
                kernel_size=3,
                strides=1,
                name=f"conv2d_{str(conv_index)}"
            )
            # batch normalization
            # layer: 输入的四维 tensor,
            # name="batch_normalization_" + str(conv_index): batchnorm 层的名字
            # training=training: 是否为训练过程
            # norm_decay=norm_decay: 在预测时计算 moving average 时的衰减率
            # norm_epsilon=norm_epsilon: 方差加上极小的数，防止除以 0 的情况
            layer = self.batch_normalization_layer(
                layer,
                name=f"batch_normalization_{str(conv_index)}",
                training=training,
                norm_decay=norm_decay,
                norm_epsilon=norm_epsilon
            )
            # 自增卷积索引
            conv_index += 1
            # 将残差结果与原始输入相加
            layer += shortcut
        # 返回经过残差卷积处理后的结果和 conv_index
        return layer, conv_index

    # ---------------------------------------#
    #   构建yolo3使用的darknet53网络结构
    # ---------------------------------------#
    # inputs: 模型输入变量conv_index: 卷积层数序号,方便根据名字加载预训练权重,weights_dict: 预训练权重
    # training: 是否为训练,norm_decay: 在预测时计算movingaverage时的衰减率,norm_epsilon: 方差加上极小的数，防止除以0的情况
    # 返回
    # conv: 经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
    # route1: 返回第26层卷积计算结果52x52x256, 供后续使用
    # route2: 返回第43层卷积计算结果26x26x512, 供后续使用
    # conv_index: 卷积层计数，方便在加载预训练模型时使用
    def darknet53(self, inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        # 在 'darknet53' 变量作用域下
        with tf.variable_scope('darknet53'):
            # 416,416,3 -> 416,416,32
            # 对输入操作，使用32个卷积核，卷积核大小为3,步长为1，命名为"conv2d_"加上当前卷积层的序号
            conv = self.conv2d_layer(
                inputs,
                filters_num=32,
                kernel_size=3,
                strides=1,
                name="conv2d_" + str(conv_index)
            )
            # 对卷积操作的结果进行批归一化处理
            # 参数说明：
            # - conv: 输入的特征图，即卷积操作的输出
            # - name: 批归一化层的命名，命名规则为"batch_normalization_"加上当前卷积层的序号
            # - training: 表示当前是否为训练模式，影响批归一化中的统计计算
            # - norm_decay: 在预测时计算moving average时的衰减率
            # - norm_epsilon: 方差加上极小的数，防止除以0的情况
            conv = self.batch_normalization_layer(
                conv,
                name="batch_normalization_" + str(conv_index),
                training=training,
                norm_decay=norm_decay,
                norm_epsilon=norm_epsilon
            )

            conv_index += 1
            # 416,416,32 -> 208,208,64
            # 对卷积结果进行残差块处理
            # 参数说明：
            # - conv: 输入的特征图，即卷积操作的输出
            # - conv_index: 当前卷积层的序号
            # - filters_num: 卷积核数量，设为64
            # - blocks_num: 残差块的数量，设为1
            # - training: 表示当前是否为训练模式，影响残差块中的统计计算
            # - norm_decay: 在预测时计算moving average时的衰减率
            # - norm_epsilon: 方差加上极小的数，防止除以0的情况
            conv, conv_index = self.Residual_block(
                conv,
                conv_index=conv_index,
                filters_num=64,
                blocks_num=1,
                training=training,
                norm_decay=norm_decay,
                norm_epsilon=norm_epsilon
            )

            # 208,208,64 -> 104,104,128
            # 对卷积结果进行残差块处理
            # 参数说明：
            # - conv: 输入的特征图，即卷积操作的输出
            # - conv_index: 当前卷积层的序号
            # - filters_num: 卷积核数量，设为128
            # - blocks_num: 残差块的数量，设为2
            # - training: 表示当前是否为训练模式，影响残差块中的统计计算
            # - norm_decay: 在预测时计算moving average时的衰减率
            # - norm_epsilon: 方差加上极小的数，防止除以0的情况
            conv, conv_index = self.Residual_block(
                conv,
                conv_index=conv_index,
                filters_num=128,
                blocks_num=2,
                training=training,
                norm_decay=norm_decay,
                norm_epsilon=norm_epsilon
            )
            # 104,104,128 -> 52,52,256
            # 对卷积结果进行残差块处理
            # 参数说明：
            # - conv: 输入的特征图，即卷积操作的输出
            # - conv_index: 当前卷积层的序号
            # - filters_num: 卷积核数量，设为256
            # - blocks_num: 残差块的数量，设为8
            # - training: 表示当前是否为训练模式，影响残差块中的统计计算
            # - norm_decay: 在预测时计算moving average时的衰减率
            # - norm_epsilon: 方差加上极小的数，防止除以0的情况
            conv, conv_index = self.Residual_block(
                conv,
                conv_index=conv_index,
                filters_num=256,
                blocks_num=8,
                training=training,
                norm_decay=norm_decay,
                norm_epsilon=norm_epsilon
            )
            # route1 = 52,52,256
            # 将当前卷积结果赋值给变量route1，用于后续操作
            route1 = conv
            # 52,52,256 -> 26,26,512
            # 对卷积结果进行残差块处理
            # 参数说明：
            # - conv: 输入的特征图，即卷积操作的输出
            # - conv_index: 当前卷积层的序号
            # - filters_num: 卷积核数量，设为512
            # - blocks_num: 残差块的数量，设为8
            # - training: 表示当前是否为训练模式，影响残差块中的统计计算
            # - norm_decay: 在预测时计算moving average时的衰减率
            # - norm_epsilon: 方差加上极小的数，防止除以0的情况
            conv, conv_index = self.Residual_block(
                conv,
                conv_index=conv_index,
                filters_num=512,
                blocks_num=8,
                training=training,
                norm_decay=norm_decay,
                norm_epsilon=norm_epsilon
            )
            # route2 = 26,26,512
            # 将当前卷积结果赋值给变量route2，用于后续操作
            route2 = conv
            # 26,26,512 -> 13,13,1024
            # 对卷积结果进行残差块处理
            # 参数说明：
            # - conv: 输入的特征图，即卷积操作的输出
            # - conv_index: 当前卷积层的序号
            # - filters_num: 卷积核数量，设为1024
            # - blocks_num: 残差块的数量，设为4
            # - training: 表示当前是否为训练模式，影响残差块中的统计计算
            # - norm_decay: 在预测时计算moving average时的衰减率
            # - norm_epsilon: 方差加上极小的数，防止除以0的情况

            conv, conv_index = self.Residual_block(
                conv,
                conv_index=conv_index,
                filters_num=1024,
                blocks_num=4,
                training=training,
                norm_decay=norm_decay,
                norm_epsilon=norm_epsilon
            )
            # route3 = 13,13,1024
        # 返回经过darknet53网络处理后的结果和 conv_index
        # - route1: 52x52x256 的特征层
        # - route2: 26x26x512 的特征层
        # - conv: 13x13x1024 的特征层
        # - conv_index: 卷积层计数，方便在加载预训练模型时使用
        return route1, route2, conv, conv_index

    # 输出两个网络结果
    # 第一个是进行5次卷积后，用于下一次逆卷积的，卷积过程是1X1，3X3，1X1，3X3，1X1
    # 第二个是进行5+2次卷积，作为一个特征层的，卷积过程是1X1，3X3，1X1，3X3，1X1，3X3，1X1
    # 输入
    # inputs: 输入特征
    # filters_num: 卷积核数量
    # out_filters: 最后输出层的卷积核数量
    # conv_index: 卷积层数序号,方便根据名字加载预训练权重
    # training: 是否为训练
    # norm_decay: 在预测时计算moving,average时的衰减率
    # norm_epsilon: 方差加上极小的数，防止除以0的情况
    # 返回
    # route: 返回最后一层卷积的前一层结果,conv: 返回最后一层卷积的结果,conv_index: conv层计数
    def yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay=0.99,
                   norm_epsilon=1e-3):
        # 使用1x1卷积进行特征提取
        # 参数说明：
        # - inputs: 输入特征
        # - filters_num: 卷积核数量
        # - kernel_size=1: 卷积核大小
        # - strides=1: 步长
        # - name=f"conv2d_{str(conv_index)}": 卷积层的名称
        conv = self.conv2d_layer(
            inputs=inputs,
            filters_num=filters_num,
            kernel_size=1,
            strides=1,
            name=f"conv2d_{str(conv_index)}"
        )

        # 使用Batch Normalization对卷积结果进行标准化
        # 参数说明：
        # - conv: 输入的特征
        # - name: Batch Normalization层的名称
        # - training: 是否为训练过程
        # - norm_decay: 在预测时计算moving average时的衰减率
        # - norm_epsilon: 方差加上极小的数，防止除以0的情况
        conv = self.batch_normalization_layer(
            conv,
            name=f"batch_normalization_{str(conv_index)}",
            training=training,
            norm_decay=norm_decay,
            norm_epsilon=norm_epsilon
        )
        # 卷积层计数自增
        conv_index += 1
        # 使用3x3卷积进行特征提取
        # 使用卷积操作进行特征提取
        # 参数说明：
        # - conv: 输入的特征
        # - filters_num=filters_num * 2: 卷积核数量
        # - kernel_size=3: 卷积核大小
        # - strides=1: 步长
        # - name=f"conv2d_{str(conv_index)}": 卷积层的名称
        conv = self.conv2d_layer(
            conv,
            filters_num=filters_num * 2,
            kernel_size=3,
            strides=1,
            name=f"conv2d_{str(conv_index)}"
        )
        # 使用批归一化对卷积结果进行处理
        # 参数说明：
        # - conv: 输入的特征
        # - name: 批归一化层的名称
        # - training=training: 是否为训练状态
        # - norm_decay=norm_decay: 在预测时计算moving average时的衰减率
        # - norm_epsilon=norm_epsilon: 方差加上极小的数，防止除以0的情况
        conv = self.batch_normalization_layer(
            conv,
            name=f"batch_normalization_{str(conv_index)}",
            training=training,
            norm_decay=norm_decay,
            norm_epsilon=norm_epsilon
        )
        # 卷积层计数自增
        conv_index += 1
        # 使用1x1卷积进行特征提取
        # 使用卷积层进行特征提取
        # 参数说明：
        # - inputs: 输入的特征
        # - filters_num=filters_num: 卷积核数量
        # - kernel_size=1: 卷积核大小
        # - strides=1: 卷积步长
        # - name=f"conv2d_{str(conv_index)}": 卷积层的名称
        conv = self.conv2d_layer(
            conv,
            filters_num=filters_num,
            kernel_size=1,
            strides=1,
            name=f"conv2d_{str(conv_index)}"
        )
        # 使用批量归一化层对卷积层的输出进行归一化
        # 参数说明：
        # - conv: 卷积层的输出
        # - name=f"batch_normalization_{str(conv_index)}": 批量归一化层的名称
        # - training=training: 是否为训练模式
        # - norm_decay=norm_decay: 在预测时计算moving average时的衰减率
        # - norm_epsilon=norm_epsilon: 方差加上极小的数，防止除以0的情况
        conv = self.batch_normalization_layer(
            conv,
            name=f"batch_normalization_{str(conv_index)}",
            training=training,
            norm_decay=norm_decay,
            norm_epsilon=norm_epsilon
        )
        # 卷积层计数自增
        conv_index += 1
        # 使用3x3卷积进行特征提取
        # 应用卷积层对输入进行卷积操作
        # 参数说明：
        # - conv: 输入数据
        # - filters_num=filters_num * 2: 卷积核数量，决定输出的通道数
        # - kernel_size=3: 卷积核的大小
        # - strides=1: 卷积的步长
        # - name: 卷积层的名称
        conv = self.conv2d_layer(
            conv,
            filters_num=filters_num * 2,
            kernel_size=3,
            strides=1,
            name=f"conv2d_{str(conv_index)}"
        )
        # 对经过卷积层的数据进行批量归一化操作
        # 参数说明：
        # - conv: 输入数据
        # - name=f"batch_normalization_{str(conv_index)}": 批量归一化层的名称
        # - training=training: 指定当前是否为训练模式
        # - norm_decay=norm_decay: 归一化衰减系数
        # - norm_epsilon=norm_epsilon: 归一化的小数值常数，防止除数为零
        conv = self.batch_normalization_layer(
            conv,
            name=f"batch_normalization_{str(conv_index)}",
            training=training,
            norm_decay=norm_decay,
            norm_epsilon=norm_epsilon
        )
        # 卷积层计数自增
        conv_index += 1
        # 应用1x1卷积操作对数据进行特征提取
        # 参数说明：
        # conv: 输入数据
        # - filters_num=filters_num: 卷积核数量（输出通道数）
        # - kernel_size=1: 卷积核大小
        # - strides=1: 卷积步长
        # - name=f"conv2d_{str(conv_index)}": 卷积层的名称
        conv = self.conv2d_layer(
            conv,
            filters_num=filters_num,
            kernel_size=1,
            strides=1,
            name=f"conv2d_{str(conv_index)}"
        )
        # 应用批量归一化操作，用于规范化卷积层的输出
        # 参数说明：111111
        # conv: 输入数据
        # - name=f"batch_normalization_{str(conv_index)}": 批量归一化层的名称
        # - training=training: 是否处于训练模式
        # - norm_decay=norm_decay: 归一化衰减系数
        # - norm_epsilon=norm_epsilon: 归一化的小数部分阈值
        conv = self.batch_normalization_layer(
            conv,
            name=f"batch_normalization_{str(conv_index)}",
            training=training,
            norm_decay=norm_decay,
            norm_epsilon=norm_epsilon
        )

        # 卷积层计数自增
        conv_index += 1
        # 将当前卷积层的输出作为路由（route）保存
        route = conv
        # 使用3x3卷积进行特征提取
        # 参数说明：
        # - conv: 输入数据
        # - filters_num=filters_num * 2: 卷积核数量（输出通道数）
        # - kernel_size=3: 卷积核大小
        # - strides=1: 卷积步长
        # - name=f"conv2d_{str(conv_index)}": 卷积层的名称
        conv = self.conv2d_layer(
            conv,
            filters_num=filters_num * 2,
            kernel_size=3,
            strides=1,
            name=f"conv2d_{str(conv_index)}"
        )
        # 应用批量归一化操作，用于规范化卷积层的输出
        # 参数说明：
        # - conv: 输入数据
        # - name=f"batch_normalization_{str(conv_index)}": 批量归一化层的名称
        # - training=training: 是否处于训练模式
        # - norm_decay=norm_decay: 归一化衰减系数
        # - norm_epsilon=norm_epsilon: 归一化的小数部分阈值
        conv = self.batch_normalization_layer(
            conv,
            name=f"batch_normalization_{str(conv_index)}",
            training=training,
            norm_decay=norm_decay,
            norm_epsilon=norm_epsilon
        )
        # 卷积层计数自增
        conv_index += 1
        # 1x1卷积，最后输出层
        # 应用卷积操作，用于提取特征
        # 参数说明：
        # - conv: 输入数据
        # - filters_num=out_filters: 卷积核的数量，控制输出的通道数
        # - kernel_size=1: 卷积核的大小
        # - strides=1: 卷积的步长
        # - name=f"conv2d_{str(conv_index)}": 卷积层的名称
        # - use_bias=True: 是否使用偏置项
        conv = self.conv2d_layer(
            conv,
            filters_num=out_filters,
            kernel_size=1,
            strides=1,
            name=f"conv2d_{str(conv_index)}",
            use_bias=True
        )
        # 卷积层计数自增
        conv_index += 1
        # 返回经过一系列卷积、批量归一化等操作后的结果
        # 参数说明：
        # - route: 中间结果，用于后续跳跃连接
        # - conv: 经过卷积操作后的特征图
        # - conv_index: 下一个卷积层的索引
        return route, conv, conv_index

    # 返回三个特征层的内容,
    # 构建yolo模型结构
    # inputs: 模型的输入变量,
    # num_anchors: 每个gridcell负责检测的anchor数量,
    # num_classes: 类别数量,
    # training: 是否为训练模式
    # 定义目标检测神经网络推理方法
    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):
        # 初始化卷积层索引
        conv_index = 1
        # 调用darknet53方法，获取三个卷积层结果和更新后的索引
        # 参数说明：
        # - inputs: 模型的输入，通常是图像或特征图
        # - conv_index: 卷积层索引，用于标识当前卷积层的编号
        # - training=training: 控制模型是否处于训练模式，根据传入的training参数确定
        # - norm_decay=self.norm_decay: 归一化层（Batch Normalization）的权重衰减参数
        # - norm_epsilon=self.norm_epsilon: 归一化层的epsilon参数，用于数值稳定性
        conv2d_26, conv2d_43, conv, conv_index = self.darknet53(
            inputs,
            conv_index,
            training=training,
            norm_decay=self.norm_decay,
            norm_epsilon=self.norm_epsilon
        )
        # 进入TensorFlow变量范围'yolo'
        with tf.variable_scope('yolo'):
            # 调用yolo_block方法，获取第一个特征层的两个卷积层结果和更新后的索引
            # 参数说明：
            # - conv: 输入特征图
            # - filters_num=512: 卷积核的数量，控制输出的通道数，此处为512
            # - out_filters=num_anchors * (num_classes + 5): 输出通道数，计算方式为num_anchors * (num_classes + 5)
            # - conv_index=conv_index: 卷积层索引，用于标识当前卷积层的编号
            # - training=training: 控制模型是否处于训练模式，根据传入的training参数确定
            # - norm_decay=self.norm_decay: 归一化层的权重衰减参数
            # - norm_epsilon=self.norm_epsilon: 归一化层的epsilon参数，用于数值稳定性
            conv2d_57, conv2d_59, conv_index = self.yolo_block(
                conv,
                filters_num=512,
                out_filters=num_anchors * (num_classes + 5),
                conv_index=conv_index,
                training=training,
                norm_decay=self.norm_decay,
                norm_epsilon=self.norm_epsilon
            )
            # 对第一个特征层的结果进行卷积和上采样操作
            # 应用卷积操作，获取第二个特征层的卷积层结果
            # 参数说明：
            # - conv2d_57: 输入特征图
            # - filters_num=256: 卷积核的数量，控制输出的通道数，此处为256
            # - kernel_size=1: 卷积核的大小，此处为1x1卷积核
            # - strides=1: 卷积的步长，此处为1
            # - name: 卷积层的名称
            conv2d_60 = self.conv2d_layer(
                conv2d_57,
                filters_num=256,
                kernel_size=1,
                strides=1,
                name=f"conv2d_{str(conv_index)}"
            )
            # 应用批量归一化操作，对第二个特征层的卷积结果进行归一化
            # 参数说明：
            # - conv2d_60: 输入特征图
            # - name=f"batch_normalization_{str(conv_index)}": 批量归一化层的名称
            # - training=training: 控制模型是否处于训练模式，根据传入的training参数确定
            # - norm_decay=self.norm_decay: 归一化层的权重衰减参数
            # - norm_epsilon=self.norm_epsilon: 归一化层的epsilon参数，用于数值稳定性
            conv2d_60 = self.batch_normalization_layer(
                conv2d_60,
                name=f"batch_normalization_{str(conv_index)}",
                training=training,
                norm_decay=self.norm_decay,
                norm_epsilon=self.norm_epsilon
            )
            # 卷积层计数自增
            conv_index += 1
            # 上采样操作:tf.image.resize_nearest_neighbor
            # 使用最近邻插值进行上采样操作，获取unSample_0
            # 参数说明：
            # - conv2d_60: 输入特征图
            # - [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]]: 上采样后的目标尺寸，为原尺寸的两倍
            # - name='upSample_0': 上采样操作的名称
            unSample_0 = tf.image.resize_nearest_neighbor(
                conv2d_60,
                size=[2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]],
                name='upSample_0'
            )
            # 将上采样结果与第二个卷积层的结果进行拼接操作
            # 参数说明：
            # - [unSample_0, conv2d_43]: 上采样结果,第二个卷积层的结果
            # - axis=-1: 拼接的轴，此处为最后一个维度
            # - name='route_0': 拼接操作的名称
            route0 = tf.concat(
                [unSample_0, conv2d_43],
                axis=-1,
                name='route_0'
            )
            # 调用yolo_block方法，获取第二个特征层的两个卷积层结果和更新后的索引
            # 参数说明：
            # - route0: 输入特征图，为上采样结果与第二个卷积层的拼接结果
            # - filters_num=256: 卷积核的数量，控制输出的通道数，此处为256
            # - out_filters=num_anchors * (num_classes + 5): 输出通道数，计算方式为num_anchors * (num_classes + 5)
            # - conv_index=conv_index: 卷积层索引，用于标识当前卷积层的编号
            # - training=training: 控制模型是否处于训练模式，根据传入的training参数确定
            # - norm_decay=self.norm_decay: 归一化层的权重衰减参数
            # - norm_epsilon=self.norm_epsilon: 归一化层的epsilon参数，用于数值稳定性
            conv2d_65, conv2d_67, conv_index = self.yolo_block(
                route0,
                filters_num=256,
                out_filters=num_anchors * (num_classes + 5),
                conv_index=conv_index,
                training=training,
                norm_decay=self.norm_decay,
                norm_epsilon=self.norm_epsilon
            )
            # 对第二个特征层的结果进行卷积操作
            # 参数说明：
            # - conv2d_65: 输入特征图，为第二个特征层的结果
            # - filters_num=128: 卷积核的数量，控制输出的通道数，此处为128
            # - kernel_size=1: 卷积核的大小，此处为1x1卷积核
            # - strides=1: 卷积的步长，此处为1
            # - name=f"conv2d_{str(conv_index)}": 卷积层的名称
            conv2d_68 = self.conv2d_layer(
                conv2d_65,
                filters_num=128,
                kernel_size=1,
                strides=1,
                name=f"conv2d_{str(conv_index)}"
            )
            # 对第二个特征层的卷积结果应用批量归一化操作
            # 参数说明：
            # - conv2d_68: 输入特征图，为第二个特征层的卷积结果
            # - name=f"batch_normalization_{str(conv_index)}": 批量归一化层的名称
            # - training=training: 控制模型是否处于训练模式，根据传入的training参数确定
            # - norm_decay=self.norm_decay: 归一化层的权重衰减参数
            # - norm_epsilon=self.norm_epsilon: 归一化层的epsilon参数，用于数值稳定性
            conv2d_68 = self.batch_normalization_layer(
                conv2d_68,
                name=f"batch_normalization_{str(conv_index)}",
                training=training,
                norm_decay=self.norm_decay,
                norm_epsilon=self.norm_epsilon
            )
            # 卷积层计数自增
            conv_index += 1
            # 上采样操作:tf.image.resize_nearest_neighbor
            # 使用最近邻插值进行上采样操作，获取unSample_1
            # 参数说明：
            # - conv2d_68: 输入特征图
            # - [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]]: 上采样后的目标尺寸，为原尺寸的两倍
            # - name: 上采样操作的名称
            unSample_1 = tf.image.resize_nearest_neighbor(
                conv2d_68,
                size=[2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]],
                name='upSample_1'
            )
            # 将上采样结果与第三个卷积层的结果进行拼接操作
            # 参数说明：
            # - [unSample_1, conv2d_26]: 上采样结果,第三个卷积层的结果
            # - axis=-1: 拼接的轴，此处为最后一个维度
            # - name='route_1': 拼接操作的名称
            route1 = tf.concat(
                [unSample_1, conv2d_26],
                axis=-1,
                name='route_1'
            )
            # 调用yolo_block方法，获取第三个特征层的两个卷积层结果
            # 参数说明：
            # - route1: 输入特征图，为上采样结果与第三个卷积层的拼接结果
            # - filters_num=128: 卷积核的数量，控制输出的通道数，此处为128
            # - out_filters=num_anchors * (num_classes + 5): 输出通道数，计算方式为num_anchors * (num_classes + 5)
            # - conv_index=conv_index: 卷积层索引，用于标识当前卷积层的编号
            # - training=training: 控制模型是否处于训练模式，根据传入的training参数确定
            # - norm_decay=self.norm_decay: 归一化层的权重衰减参数
            # - norm_epsilon=self.norm_epsilon: 归一化层的epsilon参数，用于数值稳定性
            _, conv2d_75, _ = self.yolo_block(
                route1,
                filters_num=128,
                out_filters=num_anchors * (num_classes + 5),
                conv_index=conv_index,
                training=training,
                norm_decay=self.norm_decay,
                norm_epsilon=self.norm_epsilon
            )
        # 返回三个特征层的卷积结果
        return [conv2d_59, conv2d_67, conv2d_75]
