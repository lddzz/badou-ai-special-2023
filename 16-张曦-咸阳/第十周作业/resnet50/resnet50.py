import keras
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Add, Input, ZeroPadding2D, \
    AveragePooling2D, Flatten, Dense


def identity_block(inputs_tensor, filters, stage, block):
    """
    恒等块，用于残差网络中的恒等映射。

    参数：
    x: 输入张量
    filters: 一个包含3个整数的列表，表示每个卷积层的滤波器数量

    返回：
    输出张量
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    filters1, filters2, filters3 = filters
    # 第一层卷积
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(inputs_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 第2层卷积
    x = Conv2D(filters2, (3, 3), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 第三层卷积
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = Add()([x, inputs_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, filters, stage, block, strides=(2, 2)):
    """
    卷积块，用于残差网络中的卷积映射。

    参数：
    x: 输入张量
    filters: 一个包含3个整数的列表，表示每个卷积层的滤波器数量
    stride: 卷积步幅，默认为2

    返回：
    输出张量
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    filters1, filters2, filters3 = filters

    # 第一层卷积
    x = Conv2D(filters=filters1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(
        input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 第二层卷积
    x = Conv2D(filters=filters2, kernel_size=(3, 3), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 第三层卷积
    x = Conv2D(filters=filters3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 调整 shortcut 的维度
    x_shortcut = Conv2D(filters3, (1, 1), strides=strides, padding='valid', name=conv_name_base + '1')(input_tensor)
    x_shortcut = BatchNormalization(name=bn_name_base + '1')(x_shortcut)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    image_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(image_input)

    # 预处理
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # 第一阶段
    x = conv_block(x, filters=[64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, filters=[64, 64, 256], stage=2, block='b')
    x = identity_block(x, filters=[64, 64, 256], stage=2, block='c')

    # 第二阶段
    x = conv_block(x, filters=[128, 128, 512], stage=3, block='a')
    x = identity_block(x, filters=[128, 128, 512], stage=3, block='b')
    x = identity_block(x, filters=[128, 128, 512], stage=3, block='c')
    x = identity_block(x, filters=[128, 128, 512], stage=3, block='d')

    # 第三阶段
    x = conv_block(x, filters=[256, 256, 1024], stage=4, block='a')
    x = identity_block(x, filters=[256, 256, 1024], stage=4, block='b')
    x = identity_block(x, filters=[256, 256, 1024], stage=4, block='c')
    x = identity_block(x, filters=[256, 256, 1024], stage=4, block='d')
    x = identity_block(x, filters=[256, 256, 1024], stage=4, block='e')
    x = identity_block(x, filters=[256, 256, 1024], stage=4, block='f')

    # 第四阶段
    x = conv_block(x, filters=[512, 512, 2048], stage=5, block='a')
    x = identity_block(x, filters=[512, 512, 2048], stage=5, block='b')
    x = identity_block(x, filters=[512, 512, 2048], stage=5, block='c')

    # 平均池化
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)

    # 输出层分类1000
    # x = Dense(units=1, classes=classes, activation='softmax', name='fc1000')(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    model = keras.models.Model(image_input, x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model


from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

if __name__ == '__main__':
    model = ResNet50()
    model.summary()
    # img_path = 'elephant.jpg'
    img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
