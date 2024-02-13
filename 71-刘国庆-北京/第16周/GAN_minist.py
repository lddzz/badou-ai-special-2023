# 导入Python 2.x的print函数和除法特性,使其在Python 2.x和3.x中都能正常使用
from __future__ import print_function, division
# 导入os库,用于处理文件和目录
import os
# 导入matplotlib.pyplot库,用于绘制图像和图表
import matplotlib.pyplot as plt
# 导入numpy库,用于进行科学计算
import numpy as np
# 导入tensorflow库,用于构建和训练机器学习模型
import tensorflow as tf
# 从keras.datasets导入mnist数据集
from keras.datasets import mnist
# 从keras.layers导入BatchNormalization,用于在神经网络中进行批量标准化
from keras.layers import BatchNormalization
# 从keras.layers导入Input, Dense, Reshape, Flatten,用于构建神经网络的输入、全连接层、形状重塑层和平坦层
from keras.layers import Input, Dense, Reshape, Flatten
# 从keras.layers.advanced_activations导入LeakyReLU,用于在神经网络中使用带泄露的ReLU激活函数
from keras.layers.advanced_activations import LeakyReLU
# 从keras.models导入Sequential和Model,用于构建序贯模型和通用模型
from keras.models import Sequential, Model
# 从keras.optimizers导入Adam优化器
from keras.optimizers import Adam

# 设置环境变量,控制tensorflow的日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置GPU选项,限制GPU内存使用: 限制每个GPU进程使用的显存量, 设置为"0.5"表示限制使用50%的显存
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# 创建一个新的tensorflow会话,应用上面设置的GPU选项
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# 定义GAN类
class GAN:
    # 定义初始化函数
    def __init__(self):
        # 设置图像的行数为28
        self.img_rows = 28
        # 设置图像的列数为28
        self.img_cols = 28
        # 设置图像的通道数为1
        self.img_channels = 1
        # 设置图像的形状为(28, 28, 1)
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        # 设置潜在空间的维度为100
        self.latent_dim = 100
        # 创建一个Adam优化器,学习率lr为0.0002,梯度的指数衰减率beta_1为0.5:
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        # 调用build_discriminator函数构建并编译判别器discriminator
        self.discriminator = self.build_discriminator()
        # 编译判别器,损失函数为二元交叉熵,优化器为上面创建的Adam优化器,评估指标为准确率
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        # 调用build_generator函数构建生成器generator
        self.generator = self.build_generator()
        # 创建一个输入层z,输入的形状为潜在空间的维度
        # shape=(self.latent_dim,)表示输入的形状为(self.latent_dim,)
        # 即一个长度为self.latent_dim的一维数组
        z = Input(shape=(self.latent_dim,))
        # 通过生成器生成图像
        img = self.generator(z)
        # 设置判别器为不可训练
        # 通过设置self.discriminator.trainable = False,使得判别器不可训练
        self.discriminator.trainable = False
        # 通过判别器discriminator判断生成的图像的真实性
        validity = self.discriminator(img)
        # 创建一个组合模型combined,输入为z,输出为判别器判断的真实性
        self.combined = Model(z, validity)
        # 编译组合模型,损失函数为二元交叉熵,优化器为上面创建的Adam优化器
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer
        )

    # 定义构建生成器的函数build_generator
    def build_generator(self):
        # 创建一个序贯模型
        model = Sequential()
        # 向模型中添加一个全连接层,输入维度为潜在空间的维度,输出维度为256
        model.add(Dense(256, input_dim=self.latent_dim))
        # 向模型中添加一个LeakyReLU激活层,alpha参数为0.2
        model.add(LeakyReLU(alpha=0.2))
        # 向模型中添加一个批量标准化层,动量为0.8
        model.add(BatchNormalization(momentum=0.8))
        # 向模型中添加一个全连接层,输出维度为512
        model.add(Dense(512))
        # 向模型中添加一个LeakyReLU激活层,alpha参数为0.2
        model.add(LeakyReLU(alpha=0.2))
        # 向模型中添加一个批量标准化层,动量为0.8
        model.add(BatchNormalization(momentum=0.8))
        # 向模型中添加一个全连接层,输出维度为1024
        model.add(Dense(1024))
        # 向模型中添加一个LeakyReLU激活层,alpha参数为0.2
        model.add(LeakyReLU(alpha=0.2))
        # 向模型中添加一个批量标准化层,动量为0.8
        model.add(BatchNormalization(momentum=0.8))
        # 向模型中添加一个全连接层,输出维度为图像形状的乘积,激活函数为tanh
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        # 向模型中添加一个形状重塑层,将上一层的输出重塑为图像的形状
        model.add(Reshape(self.img_shape))
        # 打印模型的结构
        model.summary()
        # 创建一个输入层noise,输入的形状为潜在空间的维度
        noise = Input(shape=(self.latent_dim,))
        # 通过模型生成图像
        img = model(noise)
        # 返回一个模型,输入为noise,输出为img
        return Model(noise, img)

    # 定义构建判别器的函数build_discriminator
    def build_discriminator(self):
        # 创建一个序贯模型
        model = Sequential()
        # 向模型中添加一个平坦层,输入的形状为图像的形状
        model.add(Flatten(input_shape=self.img_shape))
        # 向模型中添加一个全连接层,输出维度为512
        model.add(Dense(512))
        # 向模型中添加一个LeakyReLU激活层,alpha参数为0.2
        model.add(LeakyReLU(alpha=0.2))
        # 向模型中添加一个全连接层,输出维度为256
        model.add(Dense(256))
        # 向模型中添加一个LeakyReLU激活层,alpha参数为0.2
        model.add(LeakyReLU(alpha=0.2))
        # 向模型中添加一个全连接层,输出维度为1,激活函数为sigmoid
        model.add(Dense(1, activation='sigmoid'))
        # 打印模型的结构
        model.summary()
        # 创建一个输入层img,输入的形状为图像的形状
        img = Input(shape=self.img_shape)
        # 通过模型判断图像的真实性
        validity = model(img)
        # 返回一个模型,输入为img,输出为validity
        return Model(img, validity)

    # 定义训练函数,参数为训练的轮数epochs、批量大小batch_size=128和采样间隔sample_interval=50
    def train(self, epochs, batch_size=128, sample_interval=50):
        # 加载mnist数据集,(x_train, _), (_, _)只需要训练集的图像,不需要标签,测试集也不需要
        (x_train, _), (_, _) = mnist.load_data()
        # 将图像的像素值从[0, 255]缩放到[-1, 1]：(x_train - 127.5) / 127.5
        x_train = x_train / 127.5 - 1.
        # 将图像的形状从(width, height)扩展为(width, height, channels),axis=3表示在第4个维度上扩展
        x_train = np.expand_dims(x_train, axis=3)
        # 创建一个全为1的数组,形状为(batch_size, 1),表示真实的图像valid
        valid = np.ones((batch_size, 1))
        # 创建一个全为0的数组,形状为(batch_size, 1),表示生成的图像fake
        fake = np.zeros((batch_size, 1))
        # 对每一个轮次进行训练
        for epoch in range(epochs):
            # 从训练集中随机选择batch_size个图像
            # (0, x_train.shape[0])表示从0到x_train.shape[0]的范围中随机选择
            # size=batch_size表示选择batch_size个图像
            # index是一个长度为batch_size的一维数组,表示选择的图像的索引
            index = np.random.randint(0, x_train.shape[0], size=batch_size)
            # 从训练集中选择index对应的图像
            imgs = x_train[index]
            # 从标准正态分布中随机生成batch_size个噪声:均值为0,标准差为1,形状为(batch_size, self.latent_dim)
            noise = np.random.normal(loc=0, scale=1, size=(batch_size, self.latent_dim))
            # 通过生成器生成图像
            gen_imgs = self.generator.predict(noise)
            # 训练判别器,输入为真实的图像,输出为全为1的数组,返回的是损失和准确率d_loss_real
            # train_on_batch()方法是在一个batch的数据上进行一次参数更新
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            # 训练判别器,输入为生成的图像,输出为全为0的数组,返回的是损失和准确率d_loss_fake
            # train_on_batch()方法是在一个batch的数据上进行一次参数更新
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            # 计算真实的图像的损失和生成的图像的损失的 平均值
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # 从标准正态分布中随机生成batch_size个噪声,均值为0,标准差为1,形状为(batch_size, self.latent_dim)
            noise = np.random.normal(loc=0, scale=1, size=(batch_size, self.latent_dim))
            # 训练组合模型,输入为噪声,输出为全为1的数组,返回的是损失
            # train_on_batch()方法是在一个batch的数据上进行一次参数更新
            g_loss = self.combined.train_on_batch(noise, valid)
            # 打印训练的轮数、判别器的损失、判别器的准确率和生成器的损失
            if epoch % 200 == 0:
                print(
                    f"训练次数:{epoch} [判别器损失:{d_loss[0]},判别器准确率:{100 * d_loss[1]}%] [生成器的损失:{g_loss}]")
            # 如果训练次数可以被采样间隔整除,那么采样生成的图像
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # 定义采样函数sample_images,参数为当前训练次数epoch
    def sample_images(self, epoch):
        # 设置行数和列数为5
        r, c = 5, 5
        # 从标准正态分布中随机生成r*c个噪声,均值为0,标准差为1,形状为(r*c, self.latent_dim)
        noise = np.random.normal(loc=0, scale=1, size=(r * c, self.latent_dim))
        # 通过生成器生成图像
        gen_imgs = self.generator.predict(noise)
        # 将图像的像素值从[-1, 1]缩放到[0, 1]: 0.5 * gen_imgs + 0.5
        gen_imgs = 0.5 * gen_imgs + 0.5
        # 创建一个r行c列的子图
        # fig:表示整个图像,axs:表示子图
        fig, axs = plt.subplots(r, c)
        # 初始化计数器count为0
        count = 0
        # 对每一行进行操作
        for i in range(r):
            # 对每一列进行操作
            for j in range(c):
                # 在第i行第j列的子图中绘制生成的图像,颜色映射为灰度
                # gen_imgs[count, :, :, 0]表示第count个图像的像素值
                # cmap='gray'表示颜色映射为灰度
                axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                # 关闭子图的坐标轴
                axs[i, j].axis('off')
                # 计数器加1
                count += 1
        # 保存图像,文件名为"mnist_epoch.png"
        fig.savefig(f"./images/mnist_{epoch}.png")
        # 关闭图像
        plt.close()


# 主函数
if __name__ == '__main__':
    # 创建一个GAN对象
    gan = GAN()
    # 训练GAN,次数为2000,批量大小为32,采样间隔为200
    gan.train(epochs=2000, batch_size=32, sample_interval=200)
