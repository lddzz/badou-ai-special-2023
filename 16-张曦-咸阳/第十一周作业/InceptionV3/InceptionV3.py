from keras.layers import Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dense
from keras import layers
from keras.models import Model


# --------------------------------#
#   Block1 35x35
#   input_shape 35x35x192
#   output_shape 35x35x288
# --------------------------------#
def figure5(x):
    # figure5 part1
    branch1x1 = conv2D_bn(x, filters=64, kernel_size=(1, 1), strides=(1, 1))

    branch5x5 = conv2D_bn(x, filters=48, kernel_size=(1, 1), strides=(1, 1))
    branch5x5 = conv2D_bn(branch5x5, filters=64, kernel_size=(5, 5), strides=(1, 1))

    branch3x3 = conv2D_bn(x, filters=64, kernel_size=(1, 1), strides=(1, 1))
    branch3x3 = conv2D_bn(branch3x3, filters=96, kernel_size=(3, 3), strides=(1, 1))
    branch3x3 = conv2D_bn(branch3x3, filters=96, kernel_size=(3, 3), strides=(1, 1))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2D_bn(branch_pool, filters=32, kernel_size=(1, 1), strides=(1, 1))

    # 64 + 64 + 96 + 32 = 256 ，  nhwc=0123
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3, name='mixed0')

    # figure5 part2
    branch1x1 = conv2D_bn(x, filters=64, kernel_size=(1, 1), strides=(1, 1))

    branch5x5 = conv2D_bn(x, filters=48, kernel_size=(1, 1), strides=(1, 1))
    branch5x5 = conv2D_bn(branch5x5, filters=64, kernel_size=(5, 5), strides=(1, 1))

    branch3x3 = conv2D_bn(x, filters=64, kernel_size=(1, 1), strides=(1, 1))
    branch3x3 = conv2D_bn(branch3x3, filters=96, kernel_size=(3, 3), strides=(1, 1))
    branch3x3 = conv2D_bn(branch3x3, filters=96, kernel_size=(3, 3), strides=(1, 1))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2D_bn(branch_pool, filters=64, kernel_size=(1, 1), strides=(1, 1))

    # 64 +64 +96 +64 = 288， nhwc=0123
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3, name='mixed1')

    # figure5 part3
    branch1x1 = conv2D_bn(x, filters=64, kernel_size=(1, 1), strides=(1, 1))

    branch5x5 = conv2D_bn(x, filters=48, kernel_size=(1, 1), strides=(1, 1))
    branch5x5 = conv2D_bn(branch5x5, filters=64, kernel_size=(5, 5), strides=(1, 1))

    branch3x3 = conv2D_bn(x, filters=64, kernel_size=(1, 1), strides=(1, 1))
    branch3x3 = conv2D_bn(branch3x3, filters=96, kernel_size=(3, 3), strides=(1, 1))
    branch3x3 = conv2D_bn(branch3x3, filters=96, kernel_size=(3, 3), strides=(1, 1))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2D_bn(branch_pool, filters=64, kernel_size=(1, 1), strides=(1, 1))

    # 64 +64 +96 +64 = 288， nhwc=0123
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3, name='mixed2')

    return x


# --------------------------------#
#   Block2 17x17
#   input_shape 35x35x288
#   output_shape 17x17x768
# --------------------------------#
def figure6(x):
    # figure part1 : 35x35x288->17x17x768
    branch3x3 = conv2D_bn(x, filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid')

    branch3x3db1 = conv2D_bn(x, 64, kernel_size=(1, 1), strides=(1, 1))
    branch3x3db1 = conv2D_bn(branch3x3db1, 96, kernel_size=(3, 3), strides=(1, 1))
    branch3x3db1 = conv2D_bn(branch3x3db1, 96, kernel_size=(3, 3), strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3db1, branch_pool], axis=3, name='mixed3')

    # figure6 part2 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2D_bn(x, filters=192, kernel_size=(1, 1), strides=(1, 1))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2D_bn(branch_pool, filters=192, kernel_size=(1, 1), strides=(1, 1))

    branch7x7 = conv2D_bn(x, filters=128, kernel_size=(1, 1), strides=(1, 1))
    branch7x7 = conv2D_bn(branch7x7, filters=128, kernel_size=(1, 7), strides=(1, 1))
    branch7x7 = conv2D_bn(branch7x7, filters=192, kernel_size=(7, 1), strides=(1, 1))

    branch7x7db1 = conv2D_bn(x, filters=128, kernel_size=(1, 1), strides=(1, 1))
    branch7x7db1 = conv2D_bn(branch7x7db1, filters=128, kernel_size=(7, 1), strides=(1, 1))
    branch7x7db1 = conv2D_bn(branch7x7db1, filters=128, kernel_size=(1, 7), strides=(1, 1))
    branch7x7db1 = conv2D_bn(branch7x7db1, filters=128, kernel_size=(7, 1), strides=(1, 1))
    branch7x7db1 = conv2D_bn(branch7x7db1, filters=192, kernel_size=(1, 7), strides=(1, 1))

    # 192 +192 +192 +192 = 768， nhwc=0123
    x = layers.concatenate([branch1x1, branch7x7, branch7x7db1, branch_pool], axis=3, name='mixed4')

    # figure6 part3 and part4  : 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2D_bn(x, filters=192, kernel_size=(1, 1), strides=(1, 1))

        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2D_bn(branch_pool, filters=192, kernel_size=(1, 1), strides=(1, 1))

        branch7x7 = conv2D_bn(x, filters=160, kernel_size=(1, 1), strides=(1, 1))
        branch7x7 = conv2D_bn(branch7x7, filters=160, kernel_size=(1, 7), strides=(1, 1))
        branch7x7 = conv2D_bn(branch7x7, filters=192, kernel_size=(7, 1), strides=(1, 1))

        branch7x7db1 = conv2D_bn(x, filters=160, kernel_size=(1, 1), strides=(1, 1))
        branch7x7db1 = conv2D_bn(branch7x7db1, filters=160, kernel_size=(7, 1), strides=(1, 1))
        branch7x7db1 = conv2D_bn(branch7x7db1, filters=160, kernel_size=(1, 7), strides=(1, 1))
        branch7x7db1 = conv2D_bn(branch7x7db1, filters=160, kernel_size=(7, 1), strides=(1, 1))
        branch7x7db1 = conv2D_bn(branch7x7db1, filters=192, kernel_size=(1, 7), strides=(1, 1))

        # 192 +192 +192 +192 = 768， nhwc=0123
        x = layers.concatenate([branch1x1, branch7x7, branch7x7db1, branch_pool], axis=3,
                               name='mixed' + str(5 + i))

    # figure6 part5 : 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2D_bn(x, filters=192, kernel_size=(1, 1), strides=(1, 1))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2D_bn(branch_pool, filters=192, kernel_size=(1, 1), strides=(1, 1))

    branch7x7 = conv2D_bn(x, filters=192, kernel_size=(1, 1), strides=(1, 1))
    branch7x7 = conv2D_bn(branch7x7, filters=192, kernel_size=(1, 7), strides=(1, 1))
    branch7x7 = conv2D_bn(branch7x7, filters=192, kernel_size=(7, 1), strides=(1, 1))

    branch7x7db1 = conv2D_bn(x, filters=192, kernel_size=(1, 1), strides=(1, 1))
    branch7x7db1 = conv2D_bn(branch7x7db1, filters=192, kernel_size=(7, 1), strides=(1, 1))
    branch7x7db1 = conv2D_bn(branch7x7db1, filters=192, kernel_size=(1, 7), strides=(1, 1))
    branch7x7db1 = conv2D_bn(branch7x7db1, filters=192, kernel_size=(7, 1), strides=(1, 1))
    branch7x7db1 = conv2D_bn(branch7x7db1, filters=192, kernel_size=(1, 7), strides=(1, 1))

    # 192 +192 +192 +192 = 768， nhwc=0123
    x = layers.concatenate([branch1x1, branch7x7, branch7x7db1, branch_pool], axis=3, name='mixed7')
    return x


# --------------------------------#
#   Block3 17x17
#   input_shape 17x17x768
#   output_shape 8x8x2048
# --------------------------------#
def figure7(x):
    branch3x3 = conv2D_bn(x, filters=192, kernel_size=(1, 1), strides=(1, 1))
    branch3x3 = conv2D_bn(branch3x3, filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid')

    branch7x7 = conv2D_bn(x, filters=192, kernel_size=(1, 1), strides=(1, 1))
    branch7x7 = conv2D_bn(branch7x7, filters=192, kernel_size=(1, 7), strides=(1, 1))
    branch7x7 = conv2D_bn(branch7x7, filters=192, kernel_size=(7, 1), strides=(1, 1))
    branch7x7 = conv2D_bn(branch7x7, filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid')

    max_pool = MaxPooling2D(strides=(2, 2), pool_size=(3, 3))(x)
    x = layers.concatenate([branch3x3, branch7x7, max_pool], axis=3, name='mixed8')

    # part2   # part3
    for i in range(2):
        branch1x1 = conv2D_bn(x, filters=320, kernel_size=(1, 1), strides=(1, 1))

        branch3x3 = conv2D_bn(x, filters=384, kernel_size=(1, 1), strides=(1, 1))
        branch3x3_1 = conv2D_bn(branch3x3, filters=384, kernel_size=(1, 3), strides=(1, 1))
        branch3x3_2 = conv2D_bn(branch3x3, filters=384, kernel_size=(3, 1), strides=(1, 1))
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3,
                                               name="mixed9_" + str(i))

        branch3x3dbl = conv2D_bn(x, filters=448, kernel_size=(1, 1), strides=(1, 1))
        branch3x3dbl = conv2D_bn(branch3x3dbl, filters=384, kernel_size=(3, 3), strides=(1, 1))
        branch3x3dbl_1 = conv2D_bn(branch3x3dbl, filters=384, kernel_size=(1, 3), strides=(1, 1))
        branch3x3dbl_2 = conv2D_bn(branch3x3dbl, filters=384, kernel_size=(3, 1), strides=(1, 1))
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2D_bn(branch_pool, filters=192, kernel_size=(1, 1), strides=(1, 1))

        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3,  name='mixed' + str(9 + i))
    return x


def conv2D_bn(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + "_bn"
        conv_name = name + "_conv"
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation("relu", name=name)(x)
    return x


def InceptionV3(input_shape=[299, 299, 3], classes=1000):
    input_x = Input(input_shape)

    x = conv2D_bn(input_x, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x = conv2D_bn(x, filters=32, kernel_size=(3, 3), strides=(1, 1), padding="valid")
    x = conv2D_bn(x, filters=64, kernel_size=(3, 3), strides=(1, 1))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = conv2D_bn(x, filters=80, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    x = conv2D_bn(x, filters=192, kernel_size=(3, 3), strides=(1, 1), padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    print("step1:", x.shape)

    x = figure5(x)
    print("figure5 : ", x.shape)
    x = figure6(x)
    print("figure6 : ", x.shape)
    x = figure7(x)
    print("figure7 : ", x.shape)

    # 平均池化后全连接。
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    inputs = input_x
    model = Model(inputs, x, name='inception_v3')

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

if __name__ == '__main__':
    model = InceptionV3()

    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print("expand_dims", x.shape)
    x = preprocess_input(x)
    print("x.shape", x.shape)
    preds = model.predict(x)
    print("preds", preds.shape)
    print('Predicted:', decode_predictions(preds))
