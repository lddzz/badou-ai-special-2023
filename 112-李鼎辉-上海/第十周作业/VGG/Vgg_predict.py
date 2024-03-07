from Vgg_model import vgg_16
import tensorflow as tf
import numpy as np
import Vgg_train

img1 = Vgg_train.load_image('./dog.jpg')
# 对输入的图片进行resize,使其shape满足（-1，224，224，3）
inputs = tf.placeholder(tf.float32,[None,None,3])
resized_img = Vgg_train.resize_image(inputs,(224,224))

# 建立网络结构
prediction = vgg_16(resized_img)

# 载入模型
sess = tf.Session()
ckpr_filename = './vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpr_filename)

# 最后结果进行sofmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={inputs:img1})

# 打印预测结果
print("result: ")
Vgg_train.print_prob(pre[0],'./synset.txt')

