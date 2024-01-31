#Cifar10
import tensorflow as tf
import numpy as np
import time
import math
import cifar10_data

max_steps = 4000
batch_size = 100 #每代训练的数据量
num_examples_for_eval = 10000 #测试数据量
data_dir = 'Cifar_data/cifar-10-batches-bin' #数据地址

#创建控制误差的函数
def var_with_w_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        w_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='w_loss')
        tf.add_to_collection('losses',w_loss)
    return var

#使用cifar10_data文件中定义好的函数读取训练数据文件和测试数据文件
#其中训练数据集做增强处理，测试数据不做处理
image_train,labels_train = cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
image_test,label_test = cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)

#创建x和y两个palceholder，用于在训练或评估时提供输入数据和对应的标签值
#其中x的第一个参数部位None
x = tf.placeholder(tf.float32,[batch_size,24,24,3])
y = tf.placeholder(tf.int32,[batch_size])

#建立模型
#创建第一个卷积层 shape = (kh,kw,ci,co)
kernel1 = var_with_w_loss(shape = [5,5,3,64],stddev=5e-2,w1=0.0)
conv1 = tf.nn.conv2d(x,kernel1,[1,1,1,1],padding='SAME')
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1 = tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#创建第二个卷积层
kernel2=var_with_w_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
bias2=tf.Variable(tf.constant(0.1,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#拍平，因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
reshape = tf.reshape(pool2,[batch_size,-1])  #这里面的-1代表将pool2的三维结构拉直为一维结构
dim=reshape.get_shape()[1].value             #get_shape()[1].value表示获取reshape之后的第二个维度的值

#第一个全连接层
weight1=var_with_w_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

#建立第二个全连接层
weight2=var_with_w_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

#建立第三个全连接层
weight3=var_with_w_loss(shape=[192,10],stddev=1 / 192.0,w1=0.0)
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.add(tf.matmul(local4,weight3),fc_bias3)

#计算损失
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y,tf.int64))

weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss

train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
#准确率
top_k_op=tf.nn.in_top_k(result,y,1)

init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()
    #每隔100step会计算并展示当前的los，每秒钟能训练的样本数量，以及训练一个batch花费的时间
    for step in range(max_steps):
        start_time = time.time()
        image_batch,label_batch = sess.run([image_train,labels_train])
        _,loss_value = sess.run([train_op,loss],feed_dict={x:image_batch,y:label_batch})
        duration = time.time()-start_time

        if step%100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (step, loss_value, examples_per_sec, sec_per_batch))
#计算正确率
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))  # math.ceil()函数用于求整
    true_count = 0
    total_sample_count = num_batch * batch_size

    # 在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([image_test, label_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y: label_batch})
        true_count += np.sum(predictions)

    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))