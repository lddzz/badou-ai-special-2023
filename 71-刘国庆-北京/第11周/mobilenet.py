# 导入警告模块
import warnings
# 导入NumPy库,并使用别名np
import numpy as np
# 从Keras库中导入图像预处理模块
from keras.preprocessing import image
# 从Keras库中导入模型类Model
from keras.models import Model
# 从Keras库中导入各种神经网络层,用于构建深度学习模型
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
# 从Keras库中导入解码预测结果的函数decode_predictions
from keras.applications.imagenet_utils import decode_predictions
# 从Keras库中导入backend模块,并使用别名K
from keras import backend as K


# 通过调用卷积层、批量归一化层和ReLU6激活函数来构建卷积块
def conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    # 卷积层,使用ReLU6激活函数
    # 输出通道数(滤波器数量)filters
    # 卷积核大小,例如 kernel=(3, 3)
    # 使用相同的填充padding='same',保持输入和输出的尺寸相同
    # 不使用偏置项use_bias=False
    # 步幅,控制卷积的步长strides=strides
    # 层的名称为'conv1'
    x = Conv2D(filters,
               kernel_size=kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    # 批量归一化层,层的名称为'conv1_bn'
    x = BatchNormalization(name='conv1_bn')(x)
    # ReLU6激活函数
    x = Activation(relu6, name='conv1_relu')(x)
    # 返回经过卷积块处理后的输出张量
    return x


# 通过调用深度可分离卷积层、批量归一化层和ReLU6激活函数来构建深度可分离卷积块
def depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    # 深度可分离卷积层
    # 创建深度可分离卷积层,卷积核大小为(3, 3)
    # 使用相同的填充,确保输出尺寸与输入尺寸相同
    # 控制输出通道数的深度乘子
    # 控制卷积的步长
    # 不使用偏置项
    # 为该层指定名称f"conv_dw_{block_id}"
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name=f"conv_dw_{block_id}")(inputs)
    # 批量归一化层,层的名称为f"conv_dw_{block_id}_bn"
    x = BatchNormalization(name=f"conv_dw_{block_id}_bn")(x)
    # ReLU6激活函数,层的名称为f"conv_dw_{block_id}_relu"
    x = Activation(relu6, name=f"conv_dw_{block_id}_relu")(x)
    # 创建1x1卷积层,输出通道数为pointwise_conv_filters
    # 使用相同的填充,确保输出尺寸与输入尺寸相同
    # 不使用偏置项
    # 步幅为(1, 1)
    # 为该层指定名称f"conv_pw_{block_id}"
    x = Conv2D(pointwise_conv_filters,
               (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name=f"conv_pw_{block_id}")(x)
    # 批量归一化层,层的名称为f"conv_pw_{block_id}_bn"
    x = BatchNormalization(name=f"conv_pw_{block_id}_bn")(x)
    # ReLU6激活函数,层的名称为f"conv_pw_{block_id}_relu"
    x = Activation(relu6, name=f"conv_pw_{block_id}_relu")(x)
    # 返回经过卷积块处理后的输出张量
    return x


# 定义ReLU6激活函数
def relu6(x):
    # 使用Keras中的relu函数,将大于6的正数截断为6
    return K.relu(x, max_value=6)


# 定义图像预处理函数,用于将输入数据进行归一化处理
def preprocess_input(x):
    # 像素值缩放到[0, 1]之间
    x /= 255.0
    # 中心化到[-0.5, 0.5]之间
    x -= 0.5
    # 缩放到[-1, 1]之间
    x *= 2.0
    return x


# 通过调用一系列卷积块(包括深度可分离卷积块和全局平均池化层等)来构建MobileNet模型,并加载预训练权重
def MobileNet(input_shape=[224, 224, 3], depth_multiplier=1, dropout=1e-3, classes=1000):
    # 定义输入层,形状为input_shape
    img_input = Input(shape=input_shape)
    # 第一个卷积块：224,224,3 -> 112,112,32
    # 调用卷积块函数，对输入图像进行卷积操作
    # 输入图像的张量,输出通道数(滤波器数量)filters为32,卷积的步幅为(2, 2)
    x = conv_block(img_input, 32, strides=(2, 2))
    # 深度可分离卷积块：112,112,32 -> 112,112,64
    # 调用深度可分离卷积块函数，对特征图进行深度可分离卷积操作
    # 输入特征图的张量x,输出通道数为64,深度乘数depth_multiplier,块的标识符为1
    x = depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    # 深度可分离卷积块：112,112,64 -> 56,56,128
    # 调用深度可分离卷积块函数，对特征图进行深度可分离卷积操作
    # 输入特征图的张量x,输出通道数为128,深度乘数depth_multiplier,卷积的步幅为(2, 2),块的标识符为2
    x = depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    # 深度可分离卷积块：56,56,128 -> 56,56,128
    # 调用深度可分离卷积块函数，对特征图进行深度可分离卷积操作
    # 输入特征图的张量x,输出通道数为128,深度乘数depth_multiplier,块的标识符为3
    x = depthwise_conv_block(x, 128, depth_multiplier, block_id=3)
    # 深度可分离卷积块：56,56,128 -> 28,28,256
    # 调用深度可分离卷积块函数，对特征图进行深度可分离卷积操作
    # 输入特征图的张量x,输出通道数为256,深度乘数depth_multiplier,卷积的步幅为(2, 2),块的标识符为4
    x = depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    # 深度可分离卷积块：28,28,256 -> 28,28,256
    # 调用深度可分离卷积块函数，对特征图进行深度可分离卷积操作
    # 输入特征图的张量x,输出通道数为256,深度乘数depth_multiplier,块的标识符为5
    x = depthwise_conv_block(x, 256, depth_multiplier, block_id=5)
    # 深度可分离卷积块：28,28,256 -> 14,14,512
    # 调用深度可分离卷积块函数，对特征图进行深度可分离卷积操作
    # 输入特征图的张量x,输出通道数为512,深度乘数depth_multiplier,卷积的步幅为(2, 2),块的标识符为6
    x = depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)

    # 连续的深度可分离卷积块,14,14,512 -> 7,7,1024
    # 调用深度可分离卷积块函数，对特征图进行深度可分离卷积操作
    # 输入特征图的张量x,输出通道数为512,深度乘数depth_multiplier,块的标识符为7,8,9,10,11
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 深度可分离卷积块：7,7,1024 -> 1,1,1024
    # 调用深度可分离卷积块函数，对特征图进行深度可分离卷积操作
    # 输入特征图的张量x,输出通道数为1024,深度乘数depth_multiplier,卷积的步幅为(2, 2),块的标识符为12
    x = depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    # 深度可分离卷积块：1,1,1024-> 1,1,1024
    # 调用深度可分离卷积块函数，对特征图进行深度可分离卷积操作
    # 输入特征图的张量x,输出通道数为1024,深度乘数depth_multiplier,块的标识符为13
    x = depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 全局平均池化
    x = GlobalAveragePooling2D()(x)

    # 重塑形状,尺寸为(1, 1, 1024),命名为reshape_1
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    # Dropout层,命名为dropout
    x = Dropout(dropout, name='dropout')(x)
    # 1x1卷积层,输出维度为classes,使用相同的填充padding,命名为conv_preds
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    # Softmax激活函数,命名为act_softmax
    x = Activation('softmax', name='act_softmax')(x)

    # 重塑形状,尺寸为(classes,)命名为reshape_2
    x = Reshape((classes,), name='reshape_2')(x)
    # 输入为img_input,输出为x,命名为mobilenet_1_0_224_tf,构建模型
    input = img_input
    model = Model(inputs=input, outputs=x, name='mobilenet_1_0_224_tf')
    # 加载预训练权重
    model.load_weights('mobilenet_1_0_224_tf.h5')

    # 返回模型
    return model


if __name__ == '__main__':
    # 创建MobileNet模型
    model = MobileNet([224, 224, 3])

    # 加载图像并进行预处理
    # 图像路径
    img_path = 'elephant.jpg'
    # 加载图像,调整大小为 (224, 224)
    img = image.load_img(img_path, target_size=(224, 224))
    # 将图像转换为NumPy数组
    x = image.img_to_array(img)
    # 在第0轴上添加一个维度,将形状变为 (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    # 对输入图像进行预处理
    x = preprocess_input(x)
    # 打印输入图像的形状
    print(f"输入图像形状:{x.shape}")

    # 进行模型预测
    # 使用模型对输入图像进行预测，返回一个包含各类别概率的数组
    pred = model.predict(x)
    # 打印预测结果中概率最高的类别的索引
    print(f"预测结果中概率最高类别索引:{np.argmax(pred)}")

    # 打印预测结果(只显示top1)
    print(f"预测结果:{decode_predictions(pred, 1)}")
