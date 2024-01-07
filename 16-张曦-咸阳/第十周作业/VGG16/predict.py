from VGG16 import vgg16
import utils
import tensorflow as tf

img1 = utils.load_image('test_data/mouse.jpg')

inputs = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
resize_img = utils.resize_image(inputs, (224, 224))

prediction = vgg16(resize_img, is_training=False)

sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs: img1})

print("result:")
utils.print_prob(pre[0], './synset.txt')
