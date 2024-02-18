from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

'''
这段代码定义了一个简单的神经网络模型，使用了 Keras 中的 Sequential 模型，并且添加了两个全连接层（Dense 层）
512是隐藏层神经元个数，激活函数为relu，输入数据的形状是一个长度为 28*28 的一维向量
10个输出维度，激活函数Softmax，Softmax 函数可以将神经网络的输出转换为概率分布，用于多分类问题的输出
'''
network=models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

'''
optimizer='rmsprop': 这表示使用 RMSprop 优化器来训练模型。RMSprop 是一种常用的随机梯度下降优化算法的变体，它可以自适应地调整学习率，有助于加速收敛和避免梯度爆炸的问题。

loss='categorical_crossentropy': 这表示在训练过程中使用的损失函数是分类交叉熵（categorical crossentropy）。对于多类别分类问题，特别是采用 one-hot 编码标签的情况下，通常使用分类交叉熵作为损失函数。

metrics=['accuracy']: 这表示在训练和评估过程中，模型的性能指标是准确率（accuracy）。准确率是指模型在给定数据集上正确分类的样本数与总样本数之比，是衡量分类模型性能的常用指标之一。
'''
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

'''
归一化处理

'''

train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

'''
热编码
'''
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

'''
输入数据进行训练
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''

network.fit(train_images,train_labels,epochs=5,batch_size=128)

'''
测试数据输入，检查效果
'''

test_loss,test_acc=network.evaluate(test_images,test_labels,verbose=1)
print(test_loss)
print('test_acc', test_acc)


