# 该文件负责读取Cifar-10数据并对其进行数据增强预处理

import os
import tensorflow as tf

# 设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000


# 定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10Record(object):
    pass


def read_cifar10(file_queue):
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3

    image_byte = result.height * result.width * result.depth  # 图片总的像素点数
    record_bytes = label_bytes + image_byte  # 标签 + 图像总像素数
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)  # TensorFlow 中的一个类，用于创建一个读取器，该读取器用于从二进制文件中读取具有固定长度的记录。
    result.key, value = reader.read(file_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)  # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    print("lable = ", result.label)

    # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    # 这一步是将一维数据转换成3维数据
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_byte]), [result.depth, result.height, result.width])

    result.uint8_image = tf.transpose(depth_major, [1, 2, 0])
    return result


def input(data_dir, batch_size, distorted: None):
    # 这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]  # 拼接地址

    file_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(file_queue)

    reshape_img = tf.cast(read_input.uint8_image, tf.float32)  # 将转换好的图片进行tensor 的格式的图片

    num_examples_per_epoch = num_examples_pre_epoch_for_train

    if distorted != None:
        crop_img = tf.random_crop(reshape_img, [24, 24, 3])  # 用 tf.random_crop 将读入的图片（32*32*3） 裁剪为（24*24*3）

        """
        将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数 每次调用都会随机决定是否对图像进行左右翻转 
        而 tf.image.flip_left_right：这个函数是一个确定性的操作，总是将图像进行左右翻转
        """
        flipped_img = tf.image.random_flip_left_right(crop_img)

        adjust_brightness = tf.image.random_brightness(flipped_img, max_delta=0.8)  # 随机亮度调整 tf.image.random_brightness()

        adjusted_contrast = tf.image.random_contrast(adjust_brightness,lower=0.2,upper=1.8)  # 将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数

        float_images = tf.image.per_image_standardization(adjusted_contrast)  # 预处理标准化处理，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差

        float_images.set_shape([24, 24, 3])

        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)

        images_train, labels_train = tf.train.shuffle_batch([float_images, read_input.label],  # 输入的张量列表，这里包括图像和标签
                                                            batch_size=batch_size,  # 每个批次的样本数量
                                                            num_threads=16,  # 用于读取和预处理数据的线程数
                                                            capacity=min_queue_examples + 3 * batch_size, # 队列的容量，用于存储待处理的样本
                                                            min_after_dequeue=min_queue_examples)  # 队列中保留的最小样本数，确保混合(shuffling)的效果

        return images_train, tf.reshape(labels_train, [batch_size])

    else :
        """
        # resize_image_with_crop_or_pad 是 TensorFlow 中用于裁剪或填充图像大小的函数 
        mage: 输入的图像张量，可以是任意维度的图像，数据类型一般是
        tf.uint8 或 tf.float32。
        target_height: 目标图像的高度（裁剪或填充后的高度）。
        target_width: 目标图像的宽度（裁剪或填充后的宽度）。
        """
        resized_image = tf.image.resize_image_with_crop_or_pad(reshape_img, 24, 24) # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        float_image = tf.image.per_image_standardization(resized_image)  # 剪切完成以后，直接进行图片标准化操作
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)

        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)

        return images_test, tf.reshape(labels_test, [batch_size])




