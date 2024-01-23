"""
运行环境
(python3.6_tensorflow1.14) PS C:\Users\Administrator> conda list tensorflow
# packages in environment at D:\anaconda3\envs\python3.6_tensorflow1.14:
#
# Name                    Version                   Build  Channel
tensorflow                1.14.0                   pypi_0    pypi
tensorflow-estimator      1.14.0                   pypi_0    pypi
(python3.6_tensorflow1.14) PS C:\Users\Administrator> conda list keras
# packages in environment at D:\anaconda3\envs\python3.6_tensorflow1.14:
#
# Name                    Version                   Build  Channel
keras                     2.10.0                   pypi_0    pypi
keras-applications        1.0.8                    pypi_0    pypi
keras-preprocessing       1.1.2                    pypi_0    pypi
(python3.6_tensorflow1.14) PS C:\Users\Administrator> conda list python
# packages in environment at D:\anaconda3\envs\python3.6_tensorflow1.14:
#
# Name                    Version                   Build  Channel
python                    3.6.13               h3758d61_0
(python3.6_tensorflow1.14) PS C:\Users\Administrator>

"""

import math

import tensorflow as tf
import numpy as np
import cifar10_data
import time

cifar10_dir = "cifar_data/cifar-10-batches-bin"
batch_size = 100


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
    if w1 is not None:
        # tf.nn.l2_loss(var) 是 TensorFlow 中用于计算张量 var 的 L2 范数的函数
        # L2 范数是指向量中所有元素的平方和的平方根，也称为欧几里德范数。在神经网络中，L2 范数经常用于正则化模型的权重。
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)

    return var


# 使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
# 其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
image_train, labels_train = cifar10_data.input(data_dir=cifar10_dir, batch_size=batch_size, distorted=True)
test_train, labels_test = cifar10_data.input(data_dir=cifar10_dir, batch_size=batch_size, distorted=None)

# 创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
# 要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_hat = tf.placeholder(tf.int32, [batch_size])

# 创建第一个卷积层 shape=(kh,kw,ci,co)
kernel1 = variable_with_weight_loss([5, 5, 3, 64], stddev=0.05, w1=0.0)
"""
x：输入张量，通常是一个四维张量，表示输入图像的批次数、高度、宽度和通道数。
kernel1：卷积核（也称为过滤器）张量，它是用于卷积操作的权重。卷积核的形状通常为 [filter_height, filter_width, in_channels, out_channels]，其中 filter_height 和 filter_width 是卷积核的高度和宽度，in_channels 是输入通道数，out_channels 是输出通道数。
[1, 1, 1, 1]：步幅（stride）参数，指定在输入的各个维度上的滑动步幅。这里的设置 [1, 1, 1, 1] 表示在批次数、高度、宽度和通道数维度上都是步幅为 1。这意味着卷积核在这些维度上以步幅为 1 的方式进行滑动。
padding="SAME"：填充方式，这里使用 "SAME" 表示使用零填充，使得卷积的输出大小与输入的大小相同。如果使用 "VALID"，表示不使用零填充，输出的大小会根据卷积核的大小和步幅而减小。
"""
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")  # WX
bias1 = tf.Variable(tf.constant(value=0.0, shape=[64]))  # B
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))  # WX+B
pool1 = tf.nn.max_pool(relu1, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
print("x1", x)
print("kernel1", kernel1)
print("conv1.shape", conv1)
print("bias1.shape", bias1)
print("relu1.shape", relu1)
print("pool1.shape", pool1)

kernel2 = variable_with_weight_loss([5, 5, 64, 64], stddev=0.05, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

print("kernel2.shape", kernel2)
print("conv2.shape", conv2)
print("bias2.shape", bias2)
print("relu2.shape", relu2)
print("pool2.shape", pool2)

# 因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
reshape = tf.reshape(tensor=pool2, shape=[batch_size, -1])
print("reshape = ",reshape)
dim = reshape.get_shape()[1].value
print("dim = ",dim)


# 第一层全连接层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 第二层全连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc_2 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 第三层全连接层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.nn.bias_add(tf.matmul(fc_2, weight3), fc_bias3)
"""
疑问：这里fc3 原文为什么是用 tf.add？ 不用 + or tf.nn.bias_add
"""

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_hat, dtype=tf.int32))
weight_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weight_with_l2_loss

train_op = tf.train.AdamOptimizer(0.001).minimize(loss)


# 函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op = tf.nn.in_top_k(result, y_hat, 1)


# 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志中
writer = tf.summary.FileWriter('logs', tf.get_default_graph())
writer.close()


init_op = tf.global_variables_initializer()


max_steps=4000
num_examples_for_eval=10000

with tf.Session() as sess:
    sess.run(init_op)
    # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数 num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([image_train, labels_train])
        _, loss_val = sess.run([train_op, loss], feed_dict={x: image_batch, y_hat: label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            print("step %d loss %.2f   %.1f examples/sec cost %.3f"%(step, loss_val, batch_size/duration, float(duration)))

    num_batch = int(math.ceil(num_examples_for_eval/batch_size))
    total_test_num = batch_size * num_batch

    true_count=0
    for j in range(total_test_num):
        test_batch, label_test_batch = sess.run([test_train, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: test_batch, y_hat: label_test_batch})
        true_count += np.sum(predictions)

        # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count / total_test_num) * 100))