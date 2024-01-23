"""
Python深度学习：tf.multiply（）和tf.matmul（）区别

tf.multiply()是点乘
tf.multiply(x,y)是两个矩阵对应的数据相乘，所以要求x和y的shape要一样才行，例如x=[2,3],y=[2,3]

tf.matmul()是矩阵相乘
tf.matmul（x,y）中的x和y要满足矩阵的乘法规则(x的列数=y的行数)，例如x=[2,3],y=[3,4]，输出为2行4列
"""

import tensorflow as tf

x_val = [[1, 1, 1], [2, 2, 2]]
x = tf.constant(value=x_val, shape=[2, 3])

y_val = [[3, 3, 3], [4, 4, 4]]
y = tf.constant(value=y_val, shape=[2, 3])

z_val = [[3, 3], [4, 4], [5, 5]]
z = tf.constant(value=z_val, shape=[3, 2])

out_multiply = tf.multiply(x, y)
out_matmul = tf.matmul(x, z)
with tf.Session() as sess:
    # result = sess.run(out)
    print("Result multiply of out tensor:")
    print(sess.run(out_multiply))

    print("Result matmul of out tensor:")
    print(sess.run(out_multiply))


"""
bias_add  和 直接 +  有什麽区别呢
"""

"""
在 TensorFlow 中，bias_add 和直接使用 + 运算符的主要区别在于广播（broadcasting）的处理方式。

tf.nn.bias_add：

tf.nn.bias_add 是专门设计用于添加偏置（bias）的函数。它执行的是矩阵的相加操作，但会自动进行广播（broadcasting）。
如果两个张量的维度不一致，tf.nn.bias_add 会自动广播较小的张量，使其与较大的张量的维度相匹配，然后执行相加操作。
这对于神经网络中的偏置操作非常方便，因为通常情况下，偏置是一个维度较小的向量，需要与输入的每个样本进行相加。
示例："""
import tensorflow as tf

x = tf.constant([[1, 2], [3, 4]])
bias = tf.constant([5, 6])

result = tf.nn.bias_add(x, bias)

with tf.Session() as sess:
    print("result=", sess.run([result]))  # 自动将 tf.constant([5, 6])  广播成了 tf.constant([5, 6],[5, 6]) 以适应x

# 另外一个例子
# 使用 tf.nn.bias_add 添加偏置
result_bias_add = tf.nn.bias_add(x, bias)

# 使用 tf.add 直接相加
result_add = tf.add(x, bias)

# 使用运算符 + 直接相加
result_operator_add = x + bias

with tf.Session() as sess:
    print("Result using tf.nn.bias_add:")
    print(sess.run(result_bias_add))

    print("\nResult using tf.add:")
    print(sess.run(result_add))

    print("\nResult using + operator:")
    print(sess.run(result_operator_add))


