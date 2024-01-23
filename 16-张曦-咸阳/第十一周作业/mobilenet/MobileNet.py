from keras.layers import Input, Conv2D, Activation, BatchNormalization, DepthwiseConv2D, GlobalAveragePooling2D, \
    Reshape, Dropout
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import decode_predictions


def relu6(x):
    return K.relu(x, max_value=6)


def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name='conv1',
               use_bias=False)(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_relu')(x)
    return x


def depth_wise_conv(inputs, point_wise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(point_wise_conv_filters, (1, 1), padding="same", strides=(1, 1), use_bias=False,
               name='conv_%d' % block_id)(x)

    x = BatchNormalization(name='conv_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_%d_relu' % block_id)(x)

    return x


def MobileNet(inputs_shape=[224, 224, 3], depth_multiplier=1, classes=1000, dropout=1e-3):
    inputs = Input(shape=inputs_shape)

    # 224,224,3 -> 112,112,32
    x = conv_block(inputs, 32, strides=(2, 2))

    # 112, 112, 32 -> 112, 112, 64
    x = depth_wise_conv(x, 64, depth_multiplier, block_id=1)

    # 112, 112, 64 -> 56, 56, 128
    x = depth_wise_conv(x, 128, depth_multiplier, strides=(2, 2), block_id=2)

    # 56, 56, 128 -> 56, 56, 128
    x = depth_wise_conv(x, 128, depth_multiplier, block_id=3)

    # 56, 56, 128 -> 28, 28, 256
    x = depth_wise_conv(x, 256, depth_multiplier, strides=(2, 2), block_id=4)

    # 28, 28, 256 -> 28, 28, 256
    x = depth_wise_conv(x, 256, depth_multiplier, block_id=5)

    # 28, 28, 256 -> 14, 14, 512
    x = depth_wise_conv(x, 512, depth_multiplier, strides=(2, 2), block_id=6)

    # 14, 14, 512 -> 14, 14, 512
    x = depth_wise_conv(x, 512, depth_multiplier, block_id=7)
    x = depth_wise_conv(x, 512, depth_multiplier, block_id=8)
    x = depth_wise_conv(x, 512, depth_multiplier, block_id=9)
    x = depth_wise_conv(x, 512, depth_multiplier, block_id=10)
    x = depth_wise_conv(x, 512, depth_multiplier, block_id=11)

    # 14, 14, 512 -> 7, 7, 1024
    x = depth_wise_conv(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)

    # 7, 7, 512 -> 7, 7, 1024
    x = depth_wise_conv(x, 1024, depth_multiplier, block_id=13)

    # AVG POOL
    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)

    x = Reshape((1, 1, 1024), name='reshape1')(x)

    x = Dropout(dropout, name='dropout')(x)

    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)
    img_input = inputs
    model = Model(img_input, x, name='mobilenet_1_0_224_tf')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = MobileNet(inputs_shape=(224, 224, 3))

    images = image.load_img('elephant.jpg', target_size=(224, 224))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))  # 只显示top1
