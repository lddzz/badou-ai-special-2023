# -------------------------------------------------------------#
#   ResNet50的网络部分
# -------------------------------------------------------------#
# 导入将脚本同时兼容 Python 2 和 Python 3 的 __future__ 模块
from __future__ import print_function
# 从 Keras 库中导入 layers 模块
from keras import layers
# 从 Keras 库中导入 Input 模块，用于定义模型的输入
from keras.layers import Input
# 从 Keras 库中导入一系列用于构建神经网络的层
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed, Add
# 从 Keras 库中导入激活函数 Activation 和 Flatten 层
from keras.layers import Activation, Flatten
# 导入 Keras 中的 Layer 和 InputSpec 类，用于创建自定义层
from keras.engine import Layer, InputSpec
# 导入 Keras 中的初始化器和正则化器
from keras import initializers, regularizers
# 再次导入 Keras 后端（backend）模块
from keras import backend as K


# 定义一个名为 BatchNormalization 的自定义层，继承自 Keras 的 Layer 类
class BatchNormalization(Layer):

    # BatchNormalization 层的构造函数，定义了一系列参数
    def __init__(self, epsilon=1e-3, axis=-1, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None,
                 beta_regularizer=None, **kwargs):

        # 设置支持掩码（masking）
        self.supports_masking = True
        # 获取初始化 beta 的方法
        self.beta_init = initializers.get(beta_init)
        # 获取初始化 gamma 的方法
        self.gamma_init = initializers.get(gamma_init)
        # epsilon 用于防止除零错误的小数值
        self.epsilon = epsilon
        # axis 表示归一化的轴
        self.axis = axis
        # 获取 gamma 正则化器
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        # 获取 beta 正则化器
        self.beta_regularizer = regularizers.get(beta_regularizer)
        # 权重初始化
        self.initial_weights = weights
        # 调用父类（Layer）的构造函数
        super(BatchNormalization, self).__init__(**kwargs)

    # BatchNormalization 层的构建函数，定义了如何构建层
    def build(self, input_shape):
        # 设置输入规范
        self.input_spec = [InputSpec(shape=input_shape)]
        # 根据轴设置形状
        shape = (input_shape[self.axis],)
        # 添加 gamma 权重
        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=False)
        # 添加 beta 权重
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name),
                                    trainable=False)
        # 添加用于跟踪运行均值的权重
        self.running_mean = self.add_weight(shape=shape, initializer='zero',
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)
        # 添加用于跟踪运行标准差的权重
        self.running_std = self.add_weight(shape=shape, initializer='one',
                                           name='{}_running_std'.format(self.name),
                                           trainable=False)
        # 如果有初始权重，则设置
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            # 删除初始权重，确保不再使用
            del self.initial_weights
        # 设置层为已构建
        self.built = True

    # BatchNormalization 层的调用函数，定义了层的前向传播过程
    def call(self, x, mask=None):
        # 确保层已构建
        assert self.built, 'Layer must be built before being called'

        # 获取输入张量的形状
        input_shape = K.int_shape(x)

        # 计算缩减轴，用于归一化
        reduction_axes = list(range(len(input_shape)))
        # 删除用于归一化的轴
        del reduction_axes[self.axis]

        # 设置广播形状，用于处理不同形状的输入
        broadcast_shape = [1] * len(input_shape)
        # 将广播形状中指定轴的元素更新为输入张量在该轴上的维度大小
        broadcast_shape[self.axis] = input_shape[self.axis]

        # 判断是否可以简单地进行归一化，或者需要进行广播
        if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
            # 简单归一化
            x_normed = K.batch_normalization(x, self.running_mean, self.running_std, self.beta, self.gamma,
                                             epsilon=self.epsilon)
        else:
            # 需要进行广播
            # 使用Keras的reshape函数，将运行时均值在指定广播形状下进行重新形状
            broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
            # 使用Keras的reshape函数，将运行时标准差在指定广播形状下进行重新形状
            broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
            # 使用Keras的reshape函数，将beta参数在指定广播形状下进行重新形状
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            # 使用Keras的reshape函数，将gamma参数在指定广播形状下进行重新形状
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)

            # 广播归一化
            x_normed = K.batch_normalization(x, broadcast_running_mean, broadcast_running_std, broadcast_beta,
                                             broadcast_gamma, epsilon=self.epsilon)

        # 返回归一化后的张量
        return x_normed

    # 获取层的配置信息，用于保存和加载模型时的参数配置
    def get_config(self):
        # 定义配置字典，包括epsilon、axis、gamma_regularizer和beta_regularizer等参数
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None}
        # 调用父类（Layer）的get_config方法，获取基础配置信息
        base_config = super(BatchNormalization, self).get_config()
        # 合并基础配置和自定义配置，并返回结果
        return dict(list(base_config.items()) + list(config.items()))


# 定义残差块函数，用于构建恒等映射残差块
# input_tensor: 输入张量，即前一层的输出
# kernel_size: 卷积核的大小，用于定义卷积层的滤波器
# filters: 一个包含三个整数的列表，表示每个卷积层中滤波器的数量
# stage: 残差块所属的阶段，一个整数
# block: 残差块的标识，一个字符串
def identity_block(input_tensor, kernel_size, filters, stage, block):
    # 从输入filters列表中获取卷积核数量
    filters1, filters2, filters3 = filters

    # 定义卷积层和批标准化的基本名称
    # 创建卷积层的基本名称，基于给定的阶段和块的信息
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 创建批标准化层的基本名称，基于给定的阶段和块的信息
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 第一个卷积层，1x1 卷积，使用 filters1 个滤波器，命名为 conv_name_base + '2a'
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    # 批标准化层，命名为 bn_name_base + '2a'
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第二个卷积层，使用指定大小的卷积核，padding 为 'same' 表示使用零填充
    # 滤波器数量为 filters2，命名为 conv_name_base + '2b'
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # 批标准化层，命名为 bn_name_base + '2b'
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第三个卷积层，1x1 卷积，使用 filters3 个滤波器，命名为 conv_name_base + '2c'
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # 批标准化层，命名为 bn_name_base + '2c'
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 将输入张量与卷积结果相加，实现残差连接
    x = layers.add([x, input_tensor])
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 返回构建好的残差块
    return x


# 定义卷积块函数，用于构建带有卷积和短路连接的残差块
# input_tensor: 输入张量，即前一层的输出
# kernel_size: 卷积核的大小，用于定义卷积层的滤波器
# filters: 一个包含三个整数的列表，表示每个卷积层中滤波器的数量
# stage: 卷积块所属的阶段，一个整数
# block: 卷积块的标识，一个字符串
# strides: 卷积步幅的元组，默认为 (2, 2)，表示在水平和垂直方向上的步幅
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    # 从输入filters列表中获取卷积核数量
    filters1, filters2, filters3 = filters

    # 定义卷积层和批标准化的基本名称
    # 创建卷积层的基本名称，基于给定的阶段和块的信息
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 创建批标准化层的基本名称，基于给定的阶段和块的信息
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 第一个卷积层，1x1 卷积，使用 strides 进行步幅设置
    # 滤波器数量为 filters1，命名为 conv_name_base + '2a'
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    # 批标准化层，命名为 bn_name_base + '2a'
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第二个卷积层，使用指定大小的卷积核，padding 为 'same' 表示使用零填充
    # 滤波器数量为 filters2，命名为 conv_name_base + '2b'
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # 批标准化层，命名为 bn_name_base + '2b'
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第三个卷积层，1x1 卷积，使用 filters3 个滤波器，命名为 conv_name_base + '2c'
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # 批标准化层，命名为 bn_name_base + '2c'
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 短路连接，使用 1x1 卷积进行卷积操作，步幅为 strides,命名为 conv_name_base + '1'
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    # 对短路连接结果应用批标准化，其名称为 bn_name_base + '1'
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    # 将卷积层和shortcut分支的输出相加
    x = layers.add([x, shortcut])
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 返回经过卷积块处理后的输出张量
    return x


# 定义ResNet50模型的主体部分
def ResNet50(inputs):
    # 将输入张量赋给img_input
    img_input = inputs

    # 对输入进行零填充，填充量为(3, 3)
    x = ZeroPadding2D((3, 3))(img_input)
    # 第一层卷积层，使用7x7的卷积核，步长为(2, 2)，卷积核数量为64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    # 对卷积层的输出进行批归一化
    x = BatchNormalization(name='bn_conv1')(x)
    # 对卷积层的输出进行ReLU激活
    x = Activation('relu')(x)
    # 最大池化层，使用3x3的池化窗口，步长为(2, 2)，保持维度不变
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # ResNet块,阶段2
    # 第一块(标记为 'a')
    # 该卷积块包含了1个1x1的卷积层,然后是1个3x3的卷积层,最后输出通道数为256
    # 此块的标识是阶段2的块 'a',步幅为 (1, 1)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # 第二块(标记为 'b')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为64,64,256
    # 此块的标识是阶段2的块 'b'
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    # 第三块(标记为 'c')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为64,64,256
    # 此块的标识是阶段2的块 'c'
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # ResNet块,阶段3
    # 第一块(标记为 'a')
    # 该卷积块包含了1个1x1的卷积层,1个3x3的卷积层,最后输出通道数为512
    # 此块的标识是阶段3的块 'a'
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    # 第二块(标记为 'b')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为128,128,512
    # 此块的标识是阶段3的块 'b'
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    # 第三块(标记为 'c')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为128,128,512
    # 此块的标识是阶段3的块 'c'
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    # 第四块(标记为 'd')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为128,128,512
    # 此块的标识是阶段3的块 'd'
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # ResNet块,阶段4
    # 第一块(标记为 'a')
    # 该卷积块包含了1个1x1的卷积层,1个3x3的卷积层,最后输出通道数为1024
    # 此块的标识是阶段4的块 'a'
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # 第二块(标记为 'b')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为256,256,1024
    # 此块的标识是阶段4的块 'b'
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    # 第三块(标记为 'c')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为256,256,1024
    # 此块的标识是阶段4的块 'c'
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    # 第四块(标记为 'd')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为256,256,1024
    # 此块的标识是阶段4的块 'd'
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    # 第五块(标记为 'e')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为256,256,1024
    # 此块的标识是阶段4的块 'e'
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    # 第六块(标记为 'f')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为256,256,1024
    # 此块的标识是阶段4的块 'f'
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 返回经过ResNet50主体部分处理后的输出张量
    return x


# 定义标识块（TimeDistributed版本），包含残差连接
def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    # 从输入filters列表中获取卷积核数量
    nb_filter1, nb_filter2, nb_filter3 = filters

    # 根据图像数据格式设置批归一化的轴
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # 构建卷积层和批归一化层的名称
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 第一个卷积层，使用1x1卷积核，改变输入的维度
    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 第二个卷积层，使用指定大小的卷积核，保持输入的维度
    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',
                               padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 第三个卷积层，使用1x1卷积核，不改变输入的维度
    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    # 残差连接中的shortcut分支
    shortcut = input_tensor

    # 将卷积层和shortcut分支的输出相加
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    # 返回经过标识块处理后的输出张量
    return x


# 定义卷积块（TimeDistributed版本），包含残差连接
def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    # 从输入filters列表中获取卷积核数量
    nb_filter1, nb_filter2, nb_filter3 = filters

    # 根据图像数据格式设置批归一化的轴
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # 构建卷积层和批归一化层的名称
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 第一个卷积层，使用1x1卷积核，改变输入的维度
    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
                        input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 第二个卷积层，使用指定大小的卷积核，保持输入的维度
    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable,
                               kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 第三个卷积层，使用1x1卷积核，不改变输入的维度
    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c',
                        trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    # 残差连接中的shortcut分支
    shortcut = TimeDistributed(
        Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    # 将卷积层和shortcut分支的输出相加
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    # 返回经过卷积块处理后的输出张量
    return x


# 定义用于分类的卷积块（TimeDistributed版本）
def classifier_layers(x, input_shape, trainable=False):
    # 第一个卷积块，使用3x3卷积核，输出通道数为[512, 512, 2048]，步幅为(2, 2)
    x = conv_block_td(x,
                      3,
                      [512, 512, 2048],
                      stage=5,
                      block='a',
                      input_shape=input_shape,
                      strides=(2, 2),
                      trainable=trainable)
    # 三个相同的残差块（identity block），输出通道数为[512, 512, 2048]
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    # 返回经过分类卷积块处理后的输出张量
    return x


# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 定义模型的输入张量，形状为(600, 600, 3)
    inputs = Input(shape=(600, 600, 3))
    # 构建 ResNet50 模型，使用定义好的输入张量
    model = ResNet50(inputs)
    # 打印模型的摘要信息
    # model.summary()
