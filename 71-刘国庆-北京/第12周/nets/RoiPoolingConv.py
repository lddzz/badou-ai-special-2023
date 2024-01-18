# 导入Keras的神经网络层基类
from keras.engine.topology import Layer
# 导入Keras的后端模块，通常用于执行底层框架相关的操作
import keras.backend as K

# 检查Keras的后端是否为TensorFlow
if K.backend() == 'tensorflow':
    # 如果是TensorFlow，则导入TensorFlow模块
    import tensorflow as tf


# 定义一个自定义层 RoiPoolingConv，继承自 Keras 的 Layer 基类
class RoiPoolingConv(Layer):

    # 初始化方法，设置 RoiPoolingConv 层的参数,存储池化区域的大小,感兴趣区域的数量,任何额外的关键字参数
    def __init__(self, pool_size, num_rois, **kwargs):
        # 获取图像数据格式（通道顺序）并存储在 dim_ordering 中
        self.nb_channels = None
        self.dim_ordering = K.image_data_format()
        # 确保 dim_ordering 的值在 {'channels_first', 'channels_last'} 中
        assert self.dim_ordering in {'channels_first',
                                     'channels_last'}, 'dim_ordering must be in {channels_first, channels_last}'

        # 存储池化区域的大小
        self.pool_size = pool_size
        # 感兴趣区域的数量
        self.num_rois = num_rois

        # 调用基类的构造函数，传递任何额外的关键字参数
        super(RoiPoolingConv, self).__init__(**kwargs)

    # 在层被调用时构建层，获取输入形状，并提取通道数
    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    # 返回层的输出形状，其中第一个维度为不确定的值
    def compute_output_shape(self, input_shape):
        # 输出形状是一个5维张量，第一个维度为不确定的值
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    # 定义层的正向传播逻辑
    def call(self, x, mask=None):
        # 断言：确保输入是两个元素的列表（例如，图像和感兴趣区域）
        assert (len(x) == 2)

        # 从输入列表中获取图像张量
        img = x[0]
        # 从输入列表中获取感兴趣区域张量
        rois = x[1]

        # 初始化一个空列表，用于存放处理后的图像
        outputs = []

        # 遍历每个感兴趣区域
        for roi_idx in range(self.num_rois):
            # 提取当前感兴趣区域的x坐标
            x = rois[0, roi_idx, 0]
            # 提取当前感兴趣区域的y坐标
            y = rois[0, roi_idx, 1]
            # 提取当前感兴趣区域的宽度
            w = rois[0, roi_idx, 2]
            # 提取当前感兴趣区域的高度
            h = rois[0, roi_idx, 3]

            # 将x坐标转换为整数类型
            x = K.cast(x, 'int32')
            # 将y坐标转换为整数类型
            y = K.cast(y, 'int32')
            # 将宽度转换为整数类型
            w = K.cast(w, 'int32')
            # 将高度转换为整数类型
            h = K.cast(h, 'int32')

            # 调整感兴趣区域图像大小至pool_size指定的尺寸
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            # 将调整后的图像添加至outputs列表
            outputs.append(rs)

        # 将所有处理后的图像在0轴上连接起来
        final_output = K.concatenate(outputs, axis=0)
        # 重塑输出张量的形状以匹配指定的维度
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        # 重新排列输出张量的维度
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        # 返回处理后的最终输出张量
        return final_output

