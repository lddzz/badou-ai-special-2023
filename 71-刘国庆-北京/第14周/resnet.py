# resnet.py文件是用于构建ResNet（残差网络）模型的代码文件。ResNet是一种深度学习模型,主要用于图像分类和识别任务。在这个文件中,定义了构建ResNet模型的各个部分,包括：
# 1. identity_block函数：用于构建恒等映射残差块,这是ResNet的基本构成单元
# 2. conv_block函数：用于构建带有卷积和短路连接的残差块,这也是ResNet的基本构成单元
# 3. get_resnet函数：用于构建整个ResNet模型,包括多个阶段,每个阶段包含多个残差块
# 这个文件的主要作用是定义和构建ResNet模型,包括模型的各个组成部分

# 导入必要的层和模块
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add


# 定义恒等映射残差块函数identity_block,用于构建ResNet网络中的恒等映射残差块
# 参数：
# input_tensor：输入张量,即前一层的输出
# kernel_size：卷积核的大小,用于定义卷积层的滤波器
# filters：一个包含三个整数的列表,表示每个卷积层中滤波器的数量
# stage：残差块所属的阶段,一个整数
# block：残差块的标识,一个字符串
# use_bias：布尔值,表示是否在卷积层中使用偏置项,默认为True
# train_bn：布尔值,表示是否在训练过程中更新批标准化层的统计数据,默认为True
def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    # 从 filters 中获取每个卷积层的滤波器数量
    filter1, filter2, filter3 = filters
    # 创建卷积层的基本名称,基于给定的阶段和块的信息
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 创建批标准化层的基本名称,基于给定的阶段和块的信息
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 第一个卷积层,使用 filters1 个滤波器,1x1 卷积,命名为 conv_name_base + '2a',由use_bias参数决定是否在卷积中使用偏置项
    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    # 批标准化层,命名为 bn_name_base + '2a',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据,
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第二个卷积层,使用指定大小的卷积核,padding 为 'same' 表示使用零填充
    # 滤波器数量为 filters2,命名为 conv_name_base + '2b',由use_bias参数决定是否在卷积中使用偏置项
    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    # 批标准化层,命名为 bn_name_base + '2b',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第三个卷积层,1x1 卷积,使用 filters3 个滤波器,命名为 conv_name_base + '2c',由use_bias参数决定是否在卷积中使用偏置项
    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    # 批标准化层,命名为 bn_name_base + '2c',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    # 将输入张量与卷积结果相加,实现残差连接
    x = Add()([x, input_tensor])
    # 使用激活函数 'relu' 进行非线性变换,命名为'res' + str(stage) + block + '_out'
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    # 返回构建好的残差块
    return x


# 定义卷积块函数conv_block,用于构建带有卷积和短路连接的残差块
# 参数：
# input_tensor：输入张量,即前一层的输出
# kernel_size：卷积核的大小,用于定义卷积层的滤波器
# filters：一个包含三个整数的列表,表示每个卷积层中滤波器的数量
# stage：卷积块所属的阶段,一个整数
# block：卷积块的标识,一个字符串
# strides：卷积步幅的元组,默认为 (2, 2),表示在水平和垂直方向上的步幅
# use_bias：布尔值,表示是否在卷积层中使用偏置项,默认为True
# train_bn：布尔值,表示是否在训练过程中更新批标准化层的统计数据,默认为True
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    # 从 filters 中获取每个卷积层的滤波器数量
    filter1, filter2, filter3 = filters
    # 创建卷积层的基本名称,基于给定的阶段和块的信息
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 创建批标准化层的基本名称,基于给定的阶段和块的信息
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 第一个卷积层,滤波器数量为 filters1,1x1 卷积,使用 strides 进行步幅设置
    # 命名为 conv_name_base + '2a',由use_bias参数决定是否在卷积中使用偏置项
    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    # 批标准化层,命名为 bn_name_base + '2a',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第二个卷积层,滤波器数量为 filters2,使用指定大小的卷积核,padding 为 'same' 表示使用零填充
    # 命名为 conv_name_base + '2b',由use_bias参数决定是否在卷积中使用偏置项
    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    # 批标准化层,命名为 bn_name_base + '2b',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第三个卷积层,使用 filters3 个滤波器,1x1 卷积,命名为 conv_name_base + '2c',由use_bias参数决定是否在卷积中使用偏置项
    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    # 批标准化层,命名为 bn_name_base + '2c',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    # 短路连接,使用 filters3 个滤波器,使用 1x1 卷积进行卷积操作,步幅为 strides,命名为 conv_name_base + '1',由use_bias参数决定是否在卷积中使用偏置项
    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    # 对短路连接结果应用批标准化,其名称为 bn_name_base + '1',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

    # 将卷积结果与短路连接相加,实现残差连接
    x = Add()([x, shortcut])
    # 使用激活函数 'relu' 进行非线性变换,命名为'res' + str(stage) + block + '_out'
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    # 返回构建好的卷积块
    return x


# 定义ResNet模型的函数get_resnet,用于构建整个ResNet模型
# 参数：
# input_image：输入图像,是一个张量
# stage5：布尔值,表示是否构建第五阶段的残差块,默认为False
# train_bn：布尔值,表示是否在训练过程中更新批标准化层的统计数据,默认为True
def get_resnet(input_image, stage5=False, train_bn=True):
    # 对输入图像进行零填充,填充的大小为(3, 3),防止特征图尺寸缩小太快
    x = ZeroPadding2D((3, 3))(input_image)
    # ResNet块,阶段1
    # 第一层卷积操作,使用64个7x7的卷积核,步幅为2,命名为'conv1',由use_bias参数决定是否在卷积中使用偏置项
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    # 对卷积结果进行批量归一化,命名为bn_conv1,由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = BatchNormalization(name='bn_conv1')(x, training=train_bn)
    # 使用ReLU激活函数激活卷积结果
    x = Activation('relu')(x)
    # 对输入x进行最大池化操作,池化窗口大小为3x3,步幅为2,使用"same"方式进行边缘填充,将池化结果赋值给x和C1
    C1 = x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # ResNet块,阶段2
    # 第一块(标记为 'a')
    # 该卷积块包含了1个1x1的卷积层,然后是1个3x3的卷积层,最后输出通道数为256,由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    # 此块的标识是阶段2的块 'a',步幅为 (1, 1)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    # 第二块(标记为 'b')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为64,64,256
    # 此块的标识是阶段2的块 'b',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    # 第三块(标记为 'c')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为64,64,256
    # 此块的标识是阶段2的块 'b',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    # 将构建的残差块的输出赋值给x和C2
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # ResNet块,阶段3
    # 第一块(标记为 'a')
    # 该卷积块包含了1个1x1的卷积层,1个3x3的卷积层,最后输出通道数为512
    # 此块的标识是阶段3的块 'a',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    # 第二块(标记为 'b')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为128,128,512
    # 此块的标识是阶段3的块 'b',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    # 第三块(标记为 'c')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为128,128,512
    # 此块的标识是阶段3的块 'c',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    # 第四块(标记为 'd')
    # 该残差块包含了3个1x1的卷积层,滤波器数量为128,128,512
    # 此块的标识是阶段3的块 'd'
    # 将构建的残差块的输出赋值给x和C3,由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # ResNet块,阶段4
    # 第一块(标记为 'a')
    # 该卷积块包含了1个1x1的卷积层,1个3x3的卷积层,最后输出通道数为1024
    # 此块的标识是阶段4的块 'a',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    # 设置残差块的数量为22
    block_num = 22
    # 对于每一个残差块
    for i in range(block_num):
        # 构建一个恒等映射的残差块
        # 该残差块包含了3个1x1的卷积层,滤波器数量为256,256,1024
        # 此块的标识是阶段4的块 'chr(98 + i)',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
        # 将构建的残差块的输出赋值给x
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    # 将最后一个残差块的输出赋值给C4
    C4 = x
    # ResNet块,阶段5
    # 如果stage5为True
    if stage5:
        # 第一块(标记为 'a')
        # 该卷积块包含了1个1x1的卷积层,1个3x3的卷积层,最后输出通道数为2048,滤波器数量为512,512,2048
        # 此块的标识是阶段5的块 'a',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        # 第二块(标记为 'b')
        # 该残差块包含了3个1x1的卷积层,滤波器数量为512,512,2048
        # 此块的标识是阶段5的块 'b',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        # 第二块(标记为 'c')
        # 该残差块包含了3个1x1的卷积层,滤波器数量为512,512,2048
        # 此块的标识是阶段5的块 'c',由train_bn参数决定是否在训练过程中更新这个批标准化层的统计数据
        # 将构建的残差块的输出赋值给x和C5
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    # 如果stage5为False
    else:
        # 不构建第五阶段的残差块,C5被赋值为None
        C5 = None
    # 返回一个列表,包含了ResNet模型的五个阶段的输出
    # C1, C2, C3, C4, C5分别代表了ResNet模型的五个阶段的输出
    return [C1, C2, C3, C4, C5]
