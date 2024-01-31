# -------------------------------------------------------------#
#   vgg16的网络部分
# -------------------------------------------------------------#

import tensorflow as tf

# import tensorflow.contrib.slim as slim

# 创建slim对象
slim = tf.contrib.slim


def vgg16(inputs, num_classes=1000, is_training=True,
          dropout_keep_prob=0.5, spatial_squeeze=True, scope='vgg_16'):
    with tf.compat.v1.variable_scope(scope, 'vgg_16', values=[inputs]):
        # 建立Vgg16网络 参考VGG16图

        # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 接最大池化
        net = slim.max_pool2d(net, [2, 2], scope="pool1")

        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope="pool2")

        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope="pool3")

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope="pool4")

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope="pool5")

        # 利用卷积方式层代全连接层，效果等同
        net = slim.conv2d(net, 4096, [7, 7], padding="VALID", scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

        net = slim.conv2d(net, 4096, [1, 1], padding="VALID", scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

        net = slim.conv2d(inputs=net, num_outputs=num_classes, kernel_size=[1, 1], activation_fn=None,
                          normalizer_fn=None, scope='fc8')

        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')

        return net
