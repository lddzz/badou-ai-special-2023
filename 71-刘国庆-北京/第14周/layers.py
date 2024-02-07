# 这个文件(`nets/layers.py`)是一个Python脚本,
# 它定义了一些用于构建Mask R-CNN模型的关键层和函数。
# Mask R-CNN是一种用于目标检测和实例分割的深度学习模型。
# 以下是这个文件中定义的一些主要的类和函数：
# 1. `apply_box_deltas_graph`：这个函数用于计算先验框的调整参数,以优化预测的边界框。
# 2. `clip_boxes_graph`：这个函数用于将优化后的边界框裁剪到图片窗口内,防止边界框超出图片范围。
# 3. `ProposalLayer`：这个类用于将先验框转化为建议框。
# 4. `PyramidROIAlign`：这个类用于在特征层上截取内容。
# 5. `DetectionLayer`：这个类用于细化分类建议并过滤重叠部分并返回最终结果探测。
# 6. `DetectionTargetLayer`：这个类用于找到建议框的ground_truth。
# 这个文件的主要作用是提供一些用于构建和优化Mask R-CNN模型的关键组件。
# 导入 TensorFlow 和 Keras 的相关库
import tensorflow as tf
from keras.engine import Layer
import numpy as np
from utils import utils


# ----------------------------------------------------------#
#   Proposal Layer
#   该部分代码用于将先验框转化成建议框
# ----------------------------------------------------------#
# 定义一个函数 apply_box_deltas_graph,用于计算先验框的调整参数
# 这个函数用于计算先验框boxes的调整参数deltas
# 它首先计算先验框的高度、宽度和中心坐标,然后根据调整参数更新这些值,最后返回调整后的先验框坐标
def apply_box_deltas_graph(boxes, deltas):
    # 计算原始边界框的高度,
    # boxes[:, 2] 表示取所有边界框的第三个坐标值(即 y2)
    # boxes[:, 0] 表示取所有边界框的第一个坐标值(即 y1)
    # boxes[:, 2] - boxes[:, 0] 则表示计算每个边界框的高度(即 y2 - y1)
    height = boxes[:, 2] - boxes[:, 0]
    # 计算原始边界框的宽度
    # boxes[:, 3] 表示取所有边界框的第四个坐标值(即 x2)
    # boxes[:, 1] 表示取所有边界框的第二个坐标值(即 x1)
    # boxes[:, 3] - boxes[:, 1] 则表示计算每个边界框的宽度(即 x2 - x1)
    width = boxes[:, 3] - boxes[:, 1]
    # 计算原始边界框的中心点的y坐标
    # boxes[:, 0] 表示取所有边界框的第一个坐标值(即 y1)
    # height 是之前计算出的每个边界框的高度
    # boxes[:, 0] + 0.5 * height 则表示计算每个边界框的中心点的y坐标(即 y1 + 0.5 * height)
    center_y = boxes[:, 0] + 0.5 * height
    # 计算原始边界框的中心点的x坐标
    # boxes[:, 1] 表示取所有边界框的第二个坐标值(即 x1)
    # width 是之前计算出的每个边界框的宽度
    # boxes[:, 1] + 0.5 * width 则表示计算每个边界框的中心点的x坐标(即 x1 + 0.5 * width)
    center_x = boxes[:, 1] + 0.5 * width
    # 根据调整参数更新中心点的y坐标
    # deltas[:, 0] 表示取所有调整参数的第一个值
    # height 是之前计算出的每个边界框的高度
    # deltas[:, 0] * height 则表示根据调整参数和原始边界框的高度来计算中心点y坐标的调整值
    center_y += deltas[:, 0] * height
    # 根据调整参数更新中心点的x坐标
    # deltas[:, 1] 表示取所有调整参数的第二个值
    # width 是之前计算出的每个边界框的宽度
    # deltas[:, 1] * width 则表示根据调整参数和原始边界框的宽度来计算中心点x坐标的调整值
    center_x += deltas[:, 1] * width
    # 根据调整参数更新边界框的高度:将计算出的高度的调整因子乘到原始的高度上
    # deltas[:, 2] 表示取所有调整参数的第三个值
    # tf.exp(deltas[:, 2]) 则表示根据调整参数计算高度的调整因子
    height *= tf.exp(deltas[:, 2])
    # 根据调整参数更新边界框的宽度:将计算出的宽度的调整因子乘到原始的宽度上
    # deltas[:, 3] 表示取所有调整参数的第四个值
    # tf.exp(deltas[:, 3]) 则表示根据调整参数计算宽度的调整因子
    width *= tf.exp(deltas[:, 3])
    # 计算调整后的边界框的上边(y1)坐标:中心点的y坐标减去高度的一半
    # center_y 是边界框中心点的 y 坐标,height 是边界框的高度
    y1 = center_y - 0.5 * height
    # 计算调整后的边界框的左边(x1)坐标:中心点的x坐标减去宽度的一半
    # center_x 是边界框中心点的 x 坐标,width 是边界框的宽度。
    x1 = center_x - 0.5 * width
    # 计算调整后的边界框的下边(y2)坐标:上边(y1)坐标加上高度
    # y1 是边界框上边的 y 坐标,height 是边界框的高度
    y2 = y1 + height
    # 计算调整后的边界框的右边(x2)坐标:左边(x1)坐标加上宽度
    # x1 是边界框左边的 x 坐标,width 是边界框的宽度
    x2 = x1 + width
    # 将调整后的边界框的坐标堆叠成一个新的张量:将这四个坐标按照第二个维度(即每一行内部)堆叠在一起
    # y1, x1, y2, x2 分别是调整后的边界框的上边、左边、下边、右边的坐标
    # axis=1表示按照第二个维度(即每一行内部)堆叠在一起
    # name="apply_box_deltas_out"表示给这个操作命名,方便后续调试和可视化
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    # 返回调整后的边界框
    return result


# clip_boxes_graph:用于将框裁剪到指定的窗口内。
# 它首先将窗口和框分解为各自的坐标,然后将框的坐标限制在窗口内,最后返回裁剪后的框。
#     boxes: [N, (y1, x1, y2, x2)]
#     window: [4] in the form y1, x1, y2, x2
def clip_boxes_graph(boxes, window):
    # 将窗口的坐标分解为四个单独的值
    # wy1, wx1, wy2, wx2 分别表示窗口的上边(y1)、左边(x1)、下边(y2)和右边(x2)的坐标
    # tf.split(window, 4)的作用是将window沿着第0维(因为window是一维的)分割成4个子张量,每个子张量包含1个元素
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    # 将边界框的坐标分解为四个单独的值
    # y1, x1, y2, x2 分别表示边界框的上边(y1)、左边(x1)、下边(y2)和右边(x2)的坐标
    # tf.split(boxes, 4, axis=1)的作用是将boxes沿着第1维(即每一行内部)分割成4个子张量,每个子张量包含1个元素
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # 将框的坐标限制在窗口内
    # 将边界框的上边(y1)的坐标限制在窗口的上边(wy1)和下边(wy2)之间
    # tf.minimum(y1, wy2)的作用是将y1和wy2中的较小值作为新的y1
    # tf.maximum(tf.minimum(y1, wy2), wy1)的作用是将新的y1和wy1中的较大值作为最终的y1
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    # 将边界框的左边(x1)的坐标限制在窗口的左边(wx1)和右边(wx2)之间
    # tf.minimum(x1, wx2)的作用是将x1和wx2中的较小值作为新的x1
    # tf.maximum(tf.minimum(x1, wx2), wx1)的作用是将新的x1和wx1中的较大值作为最终的x1
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    # 将边界框的下边(y2)的坐标限制在窗口的上边(wy1)和下边(wy2)之间
    # tf.minimum(y2, wy2)的作用是将y2和wy2中的较小值作为新的y2
    # tf.maximum(tf.minimum(y2, wy2), wy1)的作用是将新的y2和wy1中的较大值作为最终的y2
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    # 将边界框的右边(x2)的坐标限制在窗口的左边(wx1)和右边(wx2)之间
    # tf.minimum(x2, wx2)的作用是将x2和wx2中的较小值作为新的x2
    # tf.maximum(tf.minimum(x2, wx2), wx1)的作用是将新的x2和wx1中的较大值作为最终的x2
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    # 将裁剪后的边界框的坐标堆叠在一起,形成一个新的张量clipped
    # y1, x1, y2, x2 分别是裁剪后的边界框的上边、左边、下边、右边的坐标
    # axis=1表示按照第二个维度(即每一行内部)堆叠在一起
    # name="clipped_boxes"表示给这个操作命名,方便后续调试和可视化
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    # 设置裁剪后的边界框的形状,确保其形状为[N, 4]
    # clipped.shape[0] 是裁剪后的边界框的数量(即行数)
    # 4 是每个边界框的坐标数量(即每一行的元素数)
    clipped.set_shape((clipped.shape[0], 4))
    # 返回裁剪后的框
    return clipped


# 这是一个类,用于将先验框转化为建议框。
# 它的call方法接收输入,包括先验框内部是否有物体的得分和先验框的调整参数,然后根据这些输入计算出建议框。
class ProposalLayer(Layer):
    # __init__方法用于初始化ProposalLayer类的实例
    # proposal_count: 建议框的数量,这是一个整数,表示在非极大值抑制(NMS)步骤后保留的建议框的最大数量
    # nms_threshold: 非极大值抑制阈值,这是一个浮点数,用于在非极大值抑制步骤中确定是否应抑制两个重叠框
    # config: 配置对象,包含模型的配置信息。如果为None,则使用默认配置
    # ** kwargs: 其他关键字参数,这些参数将传递给父类的构造函数
    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        # 调用父类的构造函数,传入任何额外的关键字参数
        super(ProposalLayer, self).__init__(**kwargs)
        # 将传入的config参数保存为类的属性
        self.config = config
        # 将传入的proposal_count参数保存为类的属性
        self.proposal_count = proposal_count
        # 将传入的nms_threshold参数保存为类的属性
        self.nms_threshold = nms_threshold

    # [rpn_class, rpn_bbox, anchors]
    def call(self, inputs):
        # 这行代码从输入数据中提取出得分(scores)。
        # inputs[0]是一个三维张量,其中包含了每个锚点(anchor)的背景和前景得分。
        # 通过索引[:, :, 1],我们提取出所有锚点的前景得分,即这个锚点包含目标的概率。
        scores = inputs[0][:, :, 1]
        # 这行代码从输入数据中提取出边界框回归参数(deltas)。
        # inputs[1]是一个二维张量,其中包含了每个锚点的边界框回归参数,
        # 这些参数用于调整锚点的位置和大小,使其更接近真实的目标边界框。
        deltas = inputs[1]
        # 这行代码将边界框回归参数进行标准化。
        # self.config.RPN_BBOX_STD_DEV是一个包含四个元素的列表,
        # 分别对应边界框回归参数的四个维度(中心点的y坐标、x坐标,高度和宽度)的标准差。
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # 这行代码从输入数据中提取出锚点(anchors)。`inputs[2]`是一个二维张量,其中包含了所有锚点的坐标。每个锚点由四个坐标值表示,分别是边界框的上边(y1)、左边(x1)、下边(y2)和右边(x2)。
        anchors = inputs[2]
        # 筛选出得分前6000个的框
        # self.config.PRE_NMS_LIMIT 是预设的建议框数量上限,一般设为较大的值,如6000
        # tf.shape(anchors)[1] 是实际生成的建议框数量
        # tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1]) 是取这两者中的较小值
        # 最后,这个值被赋值给 pre_nms_limit,作为非极大值抑制步骤前的建议框数量上限
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        # 获得这些框的索引
        # tf.nn.top_k函数用于获取张量scores中最大的pre_nms_limit个数,返回值是一个TopKV2对象,包含两个属性values和indices
        # values是最大的pre_nms_limit个数,indices是这些数在原张量scores中的位置
        # 参数sorted=True表示返回的结果按照值的大小进行排序
        # 参数name="top_anchors"是给这个操作命名,方便后续调试和可视化
        # 最后通过.indices获取这些最大值在原张量scores中的索引,赋值给变量ix
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        # 获得这些框的得分
        # 这里的操作是tf.gather,用于从一个张量中获取指定索引的元素。
        # [scores, ix]是传入的参数列表,它将被传递给lambda函数。
        # lambda x, y: tf.gather(x, y)是一个匿名函数,它接收两个参数x和y,然后从x中获取y索引的元素。
        # self.config.IMAGES_PER_GPU是每个GPU处理的图像数量,它决定了批处理的大小。
        # 所以,这行代码的作用是：对每个GPU处理的图像,从得分scores中获取索引ix对应的得分。
        scores = utils.batch_slice(
            [scores, ix],
            lambda x, y: tf.gather(x, y),
            self.config.IMAGES_PER_GPU
        )
        # 获得这些框的调整参数
        # 这里的操作是tf.gather,用于从一个张量中获取指定索引的元素。
        # [deltas, ix]是传入的参数列表,它将被传递给lambda函数。
        # lambda x, y: tf.gather(x, y)是一个匿名函数,它接收两个参数x和y,然后从x中获取y索引的元素。
        # self.config.IMAGES_PER_GPU是每个GPU处理的图像数量,它决定了批处理的大小。
        # 所以,这行代码的作用是：对每个GPU处理的图像,从调整参数`deltas`中获取索引`ix`对应的调整参数。
        deltas = utils.batch_slice(
            [deltas, ix],
            lambda x, y: tf.gather(x, y),
            self.config.IMAGES_PER_GPU
        )
        # 获得这些框对应的先验框
        # 这里的操作是tf.gather,用于从一个张量中获取指定索引的元素。
        # [anchors, ix]是传入的参数列表,它将被传递给lambda函数。
        # lambda a, x: tf.gather(a, x)是一个匿名函数,它接收两个参数a和x,然后从a中获取x索引的元素。
        # self.config.IMAGES_PER_GPU是每个GPU处理的图像数量,它决定了批处理的大小。
        # 所以,这行代码的作用是：对每个GPU处理的图像,从先验框`anchors`中获取索引`ix`对应的先验框。
        # 最后,这个操作的名称被设置为"pre_nms_anchors",方便后续调试和可视化。
        pre_nms_anchors = utils.batch_slice(
            [anchors, ix],
            lambda a, x: tf.gather(a, x),
            self.config.IMAGES_PER_GPU,
            names=["pre_nms_anchors"]
        )

        # [batch, N, (y1, x1, y2, x2)]
        # 对先验框进行解码
        # utils.batch_slice是一个自定义的函数,用于在批量数据上执行某个操作。
        # 这里的操作是apply_box_deltas_graph,用于根据调整参数调整先验框的位置和大小。
        # [pre_nms_anchors, deltas]是传入的参数列表,它将被传递给lambda函数。
        # lambda x, y: apply_box_deltas_graph(x, y)是一个匿名函数,它接收两个参数x和y,然后根据y调整x的位置和大小。
        # self.config.IMAGES_PER_GPU是每个GPU处理的图像数量,它决定了批处理的大小。
        # 所以,这行代码的作用是：对每个GPU处理的图像,根据调整参数`deltas`调整先验框`pre_nms_anchors`的位置和大小。
        # 最后,这个操作的名称被设置为"refined_anchors",方便后续调试和可视化。
        boxes = utils.batch_slice(
            [pre_nms_anchors, deltas],
            lambda x, y: apply_box_deltas_graph(x, y),
            self.config.IMAGES_PER_GPU,
            names=["refined_anchors"]
        )

        # 防止超出图片范围
        # 定义一个窗口,其坐标为[0, 0, 1, 1],这通常代表整个图像
        # 数据类型为32位浮点数
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        # 这里的操作是clip_boxes_graph,用于将框裁剪到指定的窗口内。
        # boxes是传入的参数,它将被传递给lambda函数。
        # lambda x: clip_boxes_graph(x, window)是一个匿名函数,它接收一个参数x,然后将x裁剪到窗口内。
        # self.config.IMAGES_PER_GPU是每个GPU处理的图像数量,它决定了批处理的大小。
        # 所以,这行代码的作用是：对每个GPU处理的图像,将调整后的先验框`boxes`裁剪到窗口内。
        # 最后,这个操作的名称被设置为"refined_anchors_clipped",方便后续调试和可视化。
        boxes = utils.batch_slice(
            boxes,
            lambda x: clip_boxes_graph(x, window),
            self.config.IMAGES_PER_GPU,
            names=["refined_anchors_clipped"]
        )

        # 非极大抑制
        # 定义一个函数nms,用于执行非极大值抑制操作
        def nms(boxes, scores):
            # tf.image.non_max_suppression函数用于执行非极大值抑制操作
            # boxes是建议框的坐标,scores是对应的得分
            # self.proposal_count是预设的建议框数量上限
            # self.nms_threshold是非极大值抑制的阈值,用于确定是否应抑制两个重叠框
            # "rpn_non_max_suppression"是给这个操作命名,方便后续调试和可视化
            # 这个函数返回的是保留下来的建议框的索引
            indices = tf.image.non_max_suppression(
                boxes,
                scores,
                self.proposal_count,
                self.nms_threshold,
                name="rpn_non_max_suppression"
            )
            # 使用tf.gather函数根据索引获取保留下来的建议框
            proposals = tf.gather(boxes, indices)
            # 如果保留下来的建议框数量达不到预设的上限,就进行填充
            # tf.maximum函数用于计算需要填充的数量
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            # 使用tf.pad函数进行填充,填充的值默认为0
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            # 返回填充后的建议框
            return proposals

        # 这里的操作是nms,即非极大值抑制。
        # [boxes, scores]是传入的参数列表,它将被传递给nms函数。
        # self.config.IMAGES_PER_GPU是每个GPU处理的图像数量,它决定了批处理的大小。
        # 所以,这行代码的作用是：对每个GPU处理的图像,执行非极大值抑制操作,并对结果进行填充。
        proposals = utils.batch_slice(
            [boxes, scores],
            nms,
            self.config.IMAGES_PER_GPU
        )
        # 返回填充后的建议框
        return proposals

    # 定义一个名为compute_output_shape的方法,它接收一个参数input_shape
    def compute_output_shape(self, input_shape):
        # 这个方法返回一个元组,表示该层输出的形状
        # None表示批量大小,它在运行时确定
        # self.proposal_count表示每个样本的提议框数量,这是在ProposalLayer类的__init__方法中设置的
        # 4表示每个提议框的坐标(y1, x1, y2, x2)
        return None, self.proposal_count, 4


# ----------------------------------------------------------#
#   ROIAlign Layer
#   利用建议框在特征层上截取内容
# ----------------------------------------------------------#

# 定义一个函数 log2_graph,它接收一个参数 x
# 这个函数的作用是计算 x 的以 2 为底的对数
# 它首先使用 TensorFlow 的 log 函数计算 x 的自然对数,然后除以 2 的自然对数,得到 x 的以 2 为底的对数
# 最后返回计算结果
def log2_graph(x):
    return tf.log(x) / tf.log(2.0)


# 定义一个函数 parse_image_meta_graph,它接收一个参数 meta
# 这个函数的作用是将 meta 里面的参数进行分割
def parse_image_meta_graph(meta):
    # 从 meta 中提取出 image_id,它是 meta 的第一列
    image_id = meta[:, 0]
    # 从 meta 中提取出 original_image_shape,它是 meta 的第二列到第四列
    original_image_shape = meta[:, 1:4]
    # 从 meta 中提取出 image_shape,它是 meta 的第五列到第七列
    image_shape = meta[:, 4:7]
    # 从 meta 中提取出 window,它是 meta 的第八列到第十一列
    # window 的形式是 (y1, x1, y2, x2),表示图像在像素中的窗口
    window = meta[:, 7:11]
    # 从 meta 中提取出 scale,它是 meta 的第十二列
    scale = meta[:, 11]
    # 从 meta 中提取出 active_class_ids,它是 meta 的第十三列及以后的所有列
    active_class_ids = meta[:, 12:]
    # 返回一个字典,包含了从 meta 中提取出的所有参数
    # 图像的 ID:image_id,放置在字典的键 image_id 下
    # 原始图像的形状:original_image_shape,放置在字典的键 original_image_shape 下
    # 图像的形状:image_shape,放置在字典的键 image_shape 下
    # 图像在像素中的窗口:window,放置在字典的键 window 下
    # 图像的缩放比例:scale,放置在字典的键 scale 下
    # 活跃的类别 ID:active_class_ids,放置在字典的键 active_class_ids 下
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids
    }


# 这是一个类,用于在特征层上截取内容。
# 它的call方法接收输入,包括建议框的位置、一些必要的图片信息和所有的特征层,然后根据这些输入计算出截取的内容。
# 定义PyramidROIAlign类,继承自Keras的Layer类
class PyramidROIAlign(Layer):
    # 初始化方法,接收一个pool_shape参数,表示池化后的输出形状
    def __init__(self, pool_shape, **kwargs):
        # 调用父类的初始化方法,可以继承父类的所有属性和方法
        super(PyramidROIAlign, self).__init__(**kwargs)
        # 将pool_shape转换为元组,并保存在实例变量self.pool_shape中
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # 输入包括建议框、图像元数据和特征图
        # 建议框的位置:inputs[0]是一个二维张量,其中包含了所有建议框的坐标
        boxes = inputs[0]
        # 图像元数据:inputs[1]是一个二维张量,
        # 其中包含了图像的 ID、原始图像的形状、图像的形状、图像在像素中的窗口、图像的缩放比例和活跃的类别 ID
        image_meta = inputs[1]
        # 特征图:inputs[2:]是一个列表,其中包含了多个特征层
        feature_maps = inputs[2:]
        # 将建议框boxes的坐标分解为y1, x1, y2, x2,
        # tf.split(boxes, 4, axis=1)的作用是将boxes沿着第1维(即每一行内部)分割成4个子张量,每个子张量包含1个元素
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        # 建议框的高度等于y2减去y1
        h = y2 - y1
        # 建议框的宽度等于x2减去x1
        w = x2 - x1
        # 从图像元数据中解析出图像的形状
        # 调用 parse_image_meta_graph 函数,输入参数是图像元数据 image_meta
        # 这个函数的作用是将图像元数据中的各项参数进行分割并以字典的形式返回
        # 其中 'image_shape' 是字典的一个键,对应的值是图像的形状
        # 使用 ['image_shape'] 提取出图像的形状
        # 图像的形状是一个列表,其中第一个元素是图像的高度和宽度,第二个元素是图像的通道数
        # 使用 [0] 提取出图像的高度和宽度,赋值给 image_shape
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # 计算图像的面积
        # image_shape[0] * image_shape[1] 是图像的高度和宽度的乘积,即图像的面积
        # tf.cast函数将图像的面积转换为浮点数类型,以便进行后续的浮点数运算
        # 最后,这个浮点数类型的图像面积被赋值给变量image_area
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # 计算建议框所在的层级
        # 计算建议框的面积(高度h乘以宽度w),然后取平方根
        # 除以224.0(一个预设的常数,通常是输入图像的大小)和图像面积的平方根的比值
        # 然后对结果取以2为底的对数,得到的值即为建议框所在的原始层级
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        # 对原始层级进行调整,使其值在2到5之间
        # 首先,对原始层级进行四舍五入并转换为整数,然后加上4
        # 然后,使用tf.maximum函数确保层级不小于2
        # 最后,使用tf.minimum函数确保层级不大于5
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # 使用tf.squeeze函数移除roi_level的第三个维度(索引为2的维度)
        roi_level = tf.squeeze(roi_level, 2)
        # 初始化池化结果和建议框到层级的映射
        pooled = []
        box_to_level = []
        # 对每个层级进行处理
        for i, level in enumerate(range(2, 6)):
            # 找到当前层级的建议框
            # 使用tf.where函数和tf.equal函数找到roi_level等于当前层级(level)的所有元素的索引
            # tf.equal函数会比较roi_level和level的每个元素,如果相等则返回True,否则返回False
            # tf.where函数会找到所有值为True的元素的索引
            # 最后将这些索引赋值给ix
            ix = tf.where(tf.equal(roi_level, level))
            # 使用tf.gather_nd函数从boxes中获取ix索引对应的元素
            # boxes是一个张量,包含了所有的建议框
            # ix是一个索引列表,包含了当前层级的建议框在boxes中的位置
            # tf.gather_nd函数会从boxes中获取ix索引对应的元素,然后返回一个新的张量
            # 最后将这个新的张量赋值给level_boxes,它包含了当前层级的所有建议框
            level_boxes = tf.gather_nd(boxes, ix)
            # 将ix添加到box_to_level列表中
            # box_to_level是一个列表,用于存储每个层级的建议框的索引
            # 这里将当前层级的建议框的索引ix添加到列表中
            box_to_level.append(ix)
            # 获得这些建议框所属的图片的索引
            # ix 是一个二维张量,其中每一行代表一个建议框,每一列代表一个属性
            # ix[:, 0] 是使用切片操作获取 ix 的第一列(即所有行的第0个元素),这些元素代表建议框所在的图片的索引
            # tf.cast 函数将这些元素转换为整数类型(tf.int32)
            # 最后将转换类型后的结果赋值给 box_indices
            box_indices = tf.cast(ix[:, 0], tf.int32)
            # 停止梯度下降
            # 使用 tf.stop_gradient 函数,它会阻止 TensorFlow 在反向传播过程中计算 level_boxes 的梯度
            # 这意味着在训练过程中,level_boxes 的值不会被更新
            level_boxes = tf.stop_gradient(level_boxes)
            # 使用 tf.stop_gradient 函数,它会阻止 TensorFlow 在反向传播过程中计算 box_indices 的梯度
            # 这意味着在训练过程中,box_indices 的值不会被更新
            box_indices = tf.stop_gradient(box_indices)
            # 对特征图进行裁剪和调整大小
            # 将裁剪和调整大小后的特征图添加到pooled列表中
            # 使用tf.image.crop_and_resize函数对特征图进行裁剪和调整大小
            # feature_maps[i]是输入的特征图,它是一个四维张量,形状为[batch_size, height, width, channels]
            # level_boxes是建议框的坐标,它是一个二维张量,形状为[num_boxes, 4],每一行代表一个建议框的坐标(y1, x1, y2, x2)
            # box_indices是建议框所在的图片的索引,它是一个一维张量,形状为[num_boxes],每个元素代表一个建议框所在的图片的索引
            # self.pool_shape是池化后的输出形状,它是一个一维张量,形状为[2],代表池化后的高度和宽度
            # method="bilinear"表示使用双线性插值的方法进行裁剪和调整大小
            # 这个函数的返回值是一个四维张量,形状为[num_boxes, pool_height, pool_width, channels],代表裁剪和调整大小后的特征图
            pooled.append(
                tf.image.crop_and_resize(
                    feature_maps[i],
                    level_boxes,
                    box_indices,
                    self.pool_shape,
                    method="bilinear"
                )
            )
        # 将所有层级的结果合并
        # pooled 是一个列表,其中包含了所有层级的裁剪和调整大小后的特征图
        # tf.concat 函数用于将多个张量沿着某个维度连接起来
        # 这里的 pooled 是一个列表,包含了所有层级的裁剪和调整大小后的特征图
        # axis=0 表示沿着第一个维度(通常是批量大小维度)进行连接
        # 这样,所有层级的特征图就被连接成了一个大的张量
        pooled = tf.concat(pooled, axis=0)
        # 将建议框到层级的映射合并
        # box_to_level 是一个列表,其中包含了所有层级的建议框到层级的映射
        # tf.concat 函数用于将多个张量沿着某个维度连接起来
        # 这里的 box_to_level 是一个列表,包含了所有层级的建议框到层级的映射
        # axis=0 表示沿着第一个维度(通常是批量大小维度)进行连接
        # 这样,所有层级的建议框到层级的映射就被连接成了一个大的张量
        box_to_level = tf.concat(box_to_level, axis=0)
        # tf.range 函数用于生成一个序列,从0开始,步长为1,长度为 box_to_level 的第一个维度的大小(即建议框的数量)
        # tf.expand_dims 函数用于在指定的维度上增加一个维度,这里在第二个维度(索引为1的维度)上增加一个维度
        # 这样,box_range 就变成了一个二维张量,每一行代表一个建议框的索引
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        # tf.cast 函数用于将张量转换为指定的数据类型,这里将 box_to_level 转换为整数类型(tf.int32)
        # tf.concat 函数用于将多个张量沿着某个维度连接起来
        # 这里将 box_to_level 和 box_range 沿着第二个维度(索引为1的维度)进行连接
        # 这样,box_to_level 就变成了一个二维张量,每一行包含了一个建议框的层级和索引
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)
        # 对建议框进行排序,使得同一张图片的建议框聚集在一起
        # box_to_level 是一个二维张量,每一行包含了一个建议框的层级和索引
        # box_to_level[:, 0] 是使用切片操作获取 box_to_level 的第一列(即所有行的第0个元素),这些元素代表建议框的层级
        # box_to_level[:, 1] 是使用切片操作获取 box_to_level 的第二列(即所有行的第1个元素),这些元素代表建议框的索引
        # 将建议框的层级乘以一个大数(这里是100000),然后加上建议框的索引,得到一个新的张量 sorting_tensor
        # 这个操作的目的是生成一个新的张量,其中每个元素的值都是唯一的,并且可以反映出建议框的层级和索引的信息
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        # tf.nn.top_k 函数用于找出张量中最大的 k 个元素及其索引
        # 这里的输入参数是 sorting_tensor 和 k=tf.shape(box_to_level)[0]
        # sorting_tensor 是上一步生成的张量,包含了所有建议框的层级和索引的信息
        # k=tf.shape(box_to_level)[0] 表示要找出的元素的数量,等于 box_to_level 的第一个维度的大小(即建议框的数量)
        # 这个函数的返回值是一个 TopKV2 对象,其中 indices 属性包含了最大元素的索引
        # 使用切片操作 [::-1] 将索引反向,即将最大的元素的索引放在最前面
        # 最后将这个反向后的索引赋值给 ix
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        # 按顺序获得图片的索引
        # ix 是一个一维张量,包含了所有建议框的排序后的索引
        # box_to_level 是一个二维张量,每一行包含了一个建议框的层级和索引
        # box_to_level[:, 2] 是使用切片操作获取 box_to_level 的第三列(即所有行的第2个元素),这些元素代表建议框所在的图片的索引
        # tf.gather 函数用于从一个张量中获取指定索引的元素
        # 这里从 box_to_level[:, 2] 中获取 ix 索引对应的元素
        # 最后将获取的元素赋值给 ix,这样 ix 就变成了一个新的一维张量,包含了所有建议框所在的图片的排序后的索引
        ix = tf.gather(box_to_level[:, 2], ix)
        # pooled 是一个四维张量,包含了所有建议框的裁剪和调整大小后的特征图
        # tf.gather 函数用于从一个张量中获取指定索引的元素
        # 这里从 pooled 中获取 ix 索引对应的元素
        # 最后将获取的元素赋值给 pooled,这样 pooled 就变成了一个新的四维张量,包含了所有建议框的裁剪和调整大小后的特征图,且特征图的顺序与 ix 中的索引顺序一致
        pooled = tf.gather(pooled, ix)
        # 重新reshape为原来的格式
        # tf.shape 函数用于获取张量的形状,返回一个一维张量,其中每个元素代表一个维度的大小
        # boxes 是一个二维张量,包含了所有建议框的坐标
        # tf.shape(boxes)[:2] 是使用切片操作获取 boxes 形状的前两个元素,这些元素代表批量大小和建议框的数量
        # pooled 是一个四维张量,包含了所有建议框的裁剪和调整大小后的特征图
        # tf.shape(pooled)[1:] 是使用切片操作获取 pooled 形状的后三个元素,这些元素代表特征图的高度、宽度和通道数
        # tf.concat 函数用于将多个张量沿着某个维度连接起来
        # 这里将 tf.shape(boxes)[:2] 和 tf.shape(pooled)[1:] 沿着第一个维度(索引为0的维度)进行连接
        # 这样,就得到了一个新的形状,其中包含了批量大小、建议框的数量、特征图的高度、宽度和通道数
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        # tf.reshape 函数用于将张量重新整形为指定的形状
        # 这里将 pooled 重新整形为 shape 指定的形状
        # 这样,pooled 就变成了一个新的五维张量,其中包含了所有建议框的裁剪和调整大小后的特征图,且特征图的形状与原来的形状一致
        pooled = tf.reshape(pooled, shape)
        # 返回池化后的特征
        return pooled

    # 计算并返回该层输出的形状
    def compute_output_shape(self, input_shape):
        # input_shape是一个元组,表示该层输入的形状
        # input_shape[0][:2]获取输入形状的前两个维度,通常是批量大小和建议框数量
        # self.pool_shape是一个元组,表示池化后的输出形状,例如(7, 7)
        # input_shape[2][-1]获取输入形状的最后一个维度,通常是特征图的通道数
        # 使用加法操作符将这些维度连接在一起,形成一个新的元组,表示该层输出的形状
        # 最后返回该元组
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


# ----------------------------------------------------------#
#   Detection Layer
#
# ----------------------------------------------------------#
# 细化分类建议并过滤重叠部分并返回最终结果探测框
def refine_detections_graph(rois, probs, deltas, window, config):
    # 使用argmax获取每个边界框最可能的类别
    # 使用 TensorFlow 的 argmax 函数
    # probs 是一个二维张量,每一行代表一个预测结果,每一列代表一个类别,元素值是对应类别的概率
    # axis=1 表示沿着第二个维度(即每一行内部)寻找最大值,也就是找出每个预测结果中概率最大的类别
    # output_type=tf.int32 表示输出的数据类型为整数
    # 执行这行代码后,class_ids 是一个一维张量,每个元素是一个预测结果中概率最大的类别的索引
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # 创建一个索引,用于从probs和deltas中获取特定类别的概率和边界框调整参数
    # 使用 TensorFlow 的 tf.range 函数生成一个从0到probs.shape[0]的序列,长度为probs的第一个维度(通常是预测结果的数量)
    # class_ids 是一个一维张量,每个元素是一个预测结果中概率最大的类别的索引
    # 使用 TensorFlow 的 tf.stack 函数将这两个一维张量堆叠在一起,形成一个二维张量
    # axis=1 表示在第二个维度(即每一行内部)进行堆叠
    # 执行这行代码后,indices 是一个二维张量,每一行包含一个预测结果的序号和该预测结果中概率最大的类别的索引
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    # 使用gather_nd获取每个边界框最可能的类别的概率
    # 使用 TensorFlow 的 gather_nd 函数
    # probs 是一个二维张量,每一行代表一个预测结果,每一列代表一个类别,元素值是对应类别的概率
    # indices 是一个二维张量,每一行包含一个预测结果的序号和该预测结果中概率最大的类别的索引
    # 执行这行代码后,class_scores 是一个一维张量,每个元素是一个预测结果中概率最大的类别的概率
    class_scores = tf.gather_nd(probs, indices)
    # 使用gather_nd获取每个边界框最可能的类别的边界框调整参数
    # 使用 TensorFlow 的 gather_nd 函数
    # deltas 是一个三维张量,每一行代表一个预测结果,每一列代表一个类别,每一深度代表一个边界框调整参数(dy, dx, log(dh), log(dw))
    # indices 是一个二维张量,每一行包含一个预测结果的序号和该预测结果中概率最大的类别的索引
    # 执行这行代码后,deltas_specific 是一个二维张量,每一行是一个预测结果中概率最大的类别的边界框调整参数
    deltas_specific = tf.gather_nd(deltas, indices)
    # 使用 apply_box_deltas_graph 函数应用边界框调整参数到 rois,得到优化后的边界框
    # rois 是原始的预测边界框
    # deltas_specific 是每个预测结果中概率最大的类别的边界框调整参数
    # config.BBOX_STD_DEV 是一个常数,用于缩放边界框调整参数
    # 这行代码的作用是将边界框调整参数应用到原始的预测边界框上,得到优化后的边界框
    refined_rois = apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)
    # 使用 clip_boxes_graph 函数将优化后的边界框裁剪到图片窗口内
    # refined_rois 是优化后的边界框
    # window 是图片窗口,通常代表整个图片
    # 这行代码的作用是将优化后的边界框裁剪到图片窗口内,防止边界框超出图片范围
    refined_rois = clip_boxes_graph(refined_rois, window)
    # 获取类别 id 大于 0 的边界框的索引,即排除背景类别的边界框
    # class_ids 是一个一维张量,每个元素是一个预测结果中概率最大的类别的索引
    # tf.where 函数用于找出满足条件的元素的索引,这里的条件是类别 id 大于 0
    # [:, 0] 是使用切片操作获取第一列(即所有行的第 0 个元素),这些元素代表满足条件的元素的索引
    # 这行代码的作用是找出非背景类别的边界框的索引
    keep = tf.where(class_ids > 0)[:, 0]
    # 如果设置了DETECTION_MIN_CONFIDENCE,则进一步筛选出概率大于DETECTION_MIN_CONFIDENCE的边界框
    # 检查是否设置了检测最小置信度
    if config.DETECTION_MIN_CONFIDENCE:
        # 找出类别得分大于等于最小置信度的边界框的索引
        # tf.where函数返回满足条件的元素的索引
        # class_scores是一个一维张量,每个元素是一个预测结果中概率最大的类别的概率
        # [:, 0]是使用切片操作获取第一列(即所有行的第0个元素),这些元素代表满足条件的元素的索引
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        # 使用tf.sets.set_intersection找出同时满足两个条件的边界框的索引
        # tf.expand_dims函数用于在指定的维度上增加一个维度
        # keep是一个一维张量,包含了非背景类别的边界框的索引
        # conf_keep是一个一维张量,包含了概率大于等于最小置信度的边界框的索引
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
        # 使用tf.sparse_tensor_to_dense将稀疏张量转换为密集张量
        # [0]是使用切片操作获取第一行(即所有列的第0个元素),这些元素代表满足条件的元素的索引
        keep = tf.sparse_tensor_to_dense(keep)[0]
    # 获取筛选后的边界框的类别id、概率和优化后的边界框
    # 使用tf.gather函数从预测的类别ID(class_ids)中获取非极大值抑制后保留的索引(keep)对应的类别ID
    # class_ids是一个一维张量,每个元素是一个预测结果中概率最大的类别的索引
    # keep是一个一维张量,包含了非极大值抑制后保留的边界框的索引
    # 执行这行代码后,pre_nms_class_ids是一个一维张量,包含了非极大值抑制后保留的边界框的类别ID
    pre_nms_class_ids = tf.gather(class_ids, keep)
    # 使用tf.gather函数从预测的类别得分(class_scores)中获取非极大值抑制后保留的索引(keep)对应的类别得分
    # class_scores是一个一维张量,每个元素是一个预测结果中概率最大的类别的概率
    # keep是一个一维张量,包含了非极大值抑制后保留的边界框的索引
    # 执行这行代码后,pre_nms_scores是一个一维张量,包含了非极大值抑制后保留的边界框的类别得分
    pre_nms_scores = tf.gather(class_scores, keep)
    # 使用tf.gather函数从优化后的边界框(refined_rois)中获取非极大值抑制后保留的索引(keep)对应的边界框
    # refined_rois是一个二维张量,每一行是一个优化后的边界框的坐标(y1, x1, y2, x2)
    # keep是一个一维张量,包含了非极大值抑制后保留的边界框的索引
    # 执行这行代码后,pre_nms_rois是一个二维张量,包含了非极大值抑制后保留的边界框的坐标
    pre_nms_rois = tf.gather(refined_rois, keep)
    # 获取筛选后的边界框的类别id的唯一值
    # 使用 TensorFlow 的 unique 函数找出 pre_nms_class_ids 中的唯一值
    # pre_nms_class_ids 是一个一维张量,包含了非极大值抑制后保留的边界框的类别ID
    # tf.unique 函数返回一个 Unique 对象,其中 y 属性包含了唯一值,idx 属性包含了原始数据中每个元素对应的唯一值的索引
    # [0] 是使用切片操作获取 y 属性(即所有唯一值)
    # 执行这行代码后,unique_pre_nms_class_ids 是一个一维张量,包含了 pre_nms_class_ids 中的唯一值
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    # 定义一个函数,用于对每个类别的边界框进行非极大值抑制
    def nms_keep_map(class_id):
        # 获取当前类别的边界框的索引
        # 获取当前类别的边界框的索引
        # tf.where函数返回满足条件的元素的索引
        # tf.equal函数用于判断pre_nms_class_ids和class_id是否相等,返回一个布尔型张量
        # [:, 0]是使用切片操作获取第一列(即所有行的第0个元素),这些元素代表满足条件的元素的索引
        # 执行这行代码后,ixs是一个一维张量,包含了当前类别的边界框的索引
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # 对当前类别的边界框进行非极大值抑制,返回保留下来的边界框的索引
        # tf.image.non_max_suppression函数用于对边界框进行非极大值抑制
        # tf.gather函数用于从一个张量中获取指定索引的元素
        # max_output_size参数表示非极大值抑制后保留的边界框的最大数量
        # iou_threshold参数表示非极大值抑制的阈值,即边界框的交并比大于这个阈值的边界框会被抑制
        # 执行这行代码后,class_keep是一个一维张量,包含了非极大值抑制后保留的边界框的索引
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD
        )
        # 将保留下来的边界框的索引转换为在原始keep中的索引
        # 执行这行代码后,class_keep是一个一维张量,包含了非极大值抑制后保留的边界框在原始keep中的索引
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # 如果保留下来的边界框数量少于DETECTION_MAX_INSTANCES,则用-1填充
        # 计算需要填充的数量
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        # 使用tf.pad函数进行填充,填充值为-1
        # [(0, gap)]表示在第一个维度(即行)的后面填充gap个-1
        # 执行这行代码后,class_keep是一个一维张量,包含了非极大值抑制后保留的边界框在原始keep中的索引,如果数量不足则用-1填充
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
        # 设置class_keep的形状,保证其第一个维度的大小为DETECTION_MAX_INSTANCES
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        # 返回非极大值抑制后保留的边界框在原始keep中的索引,如果数量不足则用-1填充
        return class_keep

    # 对每个类别的边界框进行非极大值抑制
    # 使用 TensorFlow 的 map_fn 函数对 unique_pre_nms_class_ids 中的每个元素(即每个类别的 ID)应用 nms_keep_map 函数
    # nms_keep_map 函数的作用是对每个类别的边界框进行非极大值抑制,返回保留下来的边界框的索引
    # 执行这行代码后,nms_keep 是一个二维张量,每一行是一个类别的非极大值抑制的结果
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
    # 将所有类别的非极大值抑制的结果合并
    # 使用 TensorFlow 的 reshape 函数将 nms_keep 重塑为一维张量
    # 执行这行代码后,nms_keep 是一个一维张量,包含了所有类别的非极大值抑制的结果
    nms_keep = tf.reshape(nms_keep, [-1])
    # 获取在非极大值抑制后还保留的边界框的索引
    # 使用 TensorFlow 的 where 函数找出 nms_keep 中大于 -1 的元素的索引
    # [:, 0] 是使用切片操作获取第一列(即所有行的第 0 个元素),这些元素代表满足条件的元素的索引
    # 执行这行代码后,nms_keep 是一个一维张量,包含了在非极大值抑制后还保留的边界框的索引
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 获取在非极大值抑制后还保留的边界框的索引
    # 使用 TensorFlow 的 set_intersection 函数找出 keep 和 nms_keep 中的公共元素
    # expand_dims 函数用于在指定的维度增加一个维度,这里是在第一个维度(即行)增加一个维度
    # 执行这行代码后,keep 是一个一维张量,包含了在非极大值抑制后还保留的边界框的索引
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
    # sparse_tensor_to_dense 函数用于将稀疏张量转换为密集张量
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # 如果保留的边界框数量超过DETECTION_MAX_INSTANCES,则只保留概率最高的DETECTION_MAX_INSTANCES个边界框
    # 使用 TensorFlow 的 nn.top_k 函数找出 class_scores_keep 中最大的 num_keep 个元素,返回这些元素的索引
    # 执行这行代码后,top_ids 是一个一维张量,包含了概率最高的 num_keep 个边界框的索引
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)
    # 获取最终保留的边界框的优化后的边界框、类别id和概率
    # 使用 TensorFlow 的 concat 函数将优化后的边界框、类别id和概率沿着第二个维度(即每一行内部)连接在一起
    # 执行这行代码后,detections 是一个二维张量,每一行是一个检测结果,包括一个优化后的边界框、一个类别id和一个概率
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)
    # 如果最终保留的边界框数量少于DETECTION_MAX_INSTANCES,则用0填充
    # 使用 TensorFlow 的 pad 函数在 detections 的第一个维度(即行)的后面填充 gap 个 0
    # 执行这行代码后,detections 是一个二维张量,每一行是一个检测结果,包括一个优化后的边界框、一个类别id和一个概率,如果数量不足则用 0 填充
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    # 返回最终的检测结果,每个检测结果包括一个优化后的边界框、一个类别id和一个概率
    return detections


# 定义norm_boxes_graph函数,接收边界框和形状两个参数
def norm_boxes_graph(boxes, shape):
    # 将形状分解为高度和宽度
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    # 计算缩放因子
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    # 计算偏移量
    shift = tf.constant([0., 0., 1., 1.])
    # 使用缩放因子和偏移量将边界框的坐标转换为归一化坐标
    return tf.divide(boxes - shift, scale)


# 这是一个类,用于细化分类建议并过滤重叠部分并返回最终结果探测。
# 它的call方法接收输入,包括建议框、类别概率、类别特定的边界框偏移和图片元数据,然后根据这些输入计算出最终的探测结果
class DetectionLayer(Layer):

    # 初始化函数,接收一个可选的配置对象和其他关键字参数
    def __init__(self, config=None, **kwargs):
        # 调用父类的初始化函数
        super(DetectionLayer, self).__init__(**kwargs)
        # 保存配置对象
        self.config = config

    # call方法,接收输入并进行处理
    def call(self, inputs):
        # 从输入中获取rois,mrcnn_class,mrcnn_bbox和image_meta
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # 解析图像元数据,获取图像形状和窗口,并将窗口坐标归一化
        # 调用 parse_image_meta_graph 函数解析图像元数据,返回一个字典,包含了图像的各种元数据信息
        m = parse_image_meta_graph(image_meta)
        # 从返回的字典中获取 'image_shape' 键对应的值,这是一个包含图像形状信息的列表,取第一个元素,即当前图像的形状
        image_shape = m['image_shape'][0]
        # 从返回的字典中获取 'window' 键对应的值,这是一个包含图像窗口信息的列表
        # 调用 norm_boxes_graph 函数将窗口坐标归一化,即将窗口坐标转换为相对于图像大小的比例,方便后续处理
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # 对每个批次中的项目运行检测细化图
        # 调用 utils.batch_slice 函数,该函数用于对输入数据进行批处理
        # 输入数据包括 rois(区域建议框)、mrcnn_class(每个区域的类别概率)、mrcnn_bbox(每个区域的边界框偏移)和 window(图像窗口)
        # lambda 函数用于处理每个批次的数据,它调用 refine_detections_graph 函数对每个批次的数据进行处理
        # refine_detections_graph 函数的作用是细化检测结果,包括应用边界框偏移、执行非极大值抑制、筛选出置信度高的检测结果等
        # self.config.IMAGES_PER_GPU 是每个批次的大小,即每个批次包含的图像数量
        # 执行这行代码后,detections_batch 是一个二维张量,每一行是一个检测结果,包括一个优化后的边界框、一个类别id和一个概率
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # 重塑输出,得到[batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]的形状
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    # 计算输出形状的方法
    def compute_output_shape(self, input_shape):
        # 返回None,DETECTION_MAX_INSTANCES,6的形状
        return None, self.config.DETECTION_MAX_INSTANCES, 6

    # ----------------------------------------------------------#
    #   Detection Target Layer
    #   该部分代码会输入建议框
    #   判断建议框和真实框的重合情况
    #   筛选出内部包含物体的建议框
    #   利用建议框和真实框编码
    #   调整mask的格式使得其和预测格式相同
    # ----------------------------------------------------------#


# 用于计算boxes1和boxes2的重合程度
# boxes1, boxes2: [N, (y1, x1, y2, x2)].
# 返回 [len(boxes1), len(boxes2)]
def overlaps_graph(boxes1, boxes2):
    # 将boxes1复制并重塑,使其形状与boxes2相同
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    # 将boxes2复制,使其形状与boxes1相同
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 将b1和b2分解为各自的坐标
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    # 计算重叠区域的坐标
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    # 计算重叠区域的面积
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 计算b1和b2的面积
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    # 计算并集的面积
    union = b1_area + b2_area - intersection
    # 计算交并比
    iou = intersection / union
    # 重塑iou,使其形状与boxes1和boxes2相同
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    # 返回重叠度
    return overlaps


# 这个函数用于找到建议框的ground_truth。
# 它首先移除建议框的padding部分,然后计算建议框和所有真实框的重合程度,然后根据重合程度确定正样本和负样本,
# 最后返回内部真实存在目标的建议框、每个建议框对应的类、每个建议框应该有的调整参数和每个建议框的语义分割情况。
def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    # 确保提议框的数量大于0
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    # 如果断言失败,将停止执行后续代码
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # 移除提议框中的0值
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    # 移除真实框中的0值
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    # 使用非零值的索引获取对应的类别ID
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    # 使用非零值的索引获取对应的掩码
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")
    # 获取类别ID小于0的索引
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    # 获取类别ID大于0的索引
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    # 获取人群框
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    # 获取非人群的类别ID
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    # 获取非人群的真实框
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    # 获取非人群的掩码
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)
    # 计算提议框和真实框的重叠度
    overlaps = overlaps_graph(proposals, gt_boxes)
    # 计算提议框和人群框的重叠度
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    # 获取每个提议框与人群框的最大重叠度
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    # 获取与人群框重叠度小于0.001的提议框
    no_crowd_bool = (crowd_iou_max < 0.001)
    # 获取每个提议框的最大重叠度
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 获取重叠度大于0.5的提议框
    positive_roi_bool = (roi_iou_max >= 0.5)
    # 获取正样本的索引
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 获取重叠度小于0.5且不与人群框重叠的提议框的索引
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]
    # 计算正样本的数量
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    # 随机选择正样本
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    # 获取正样本的数量
    positive_count = tf.shape(positive_indices)[0]
    # 计算负样本的数量
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    # 随机选择负样本
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # 获取正样本的提议框
    positive_rois = tf.gather(proposals, positive_indices)
    # 获取负样本的提议框
    negative_rois = tf.gather(proposals, negative_indices)
    # 获取正样本的重叠度
    positive_overlaps = tf.gather(overlaps, positive_indices)
    # 获取每个正样本的最大重叠度对应的真实框的索引
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    # 获取每个正样本的最大重叠度对应的真实框
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    # 获取每个正样本的最大重叠度对应的类别ID
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)
    # 计算正样本的提议框与对应的真实框的偏移量
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    # 将偏移量除以标准偏差进行标准化
    deltas /= config.BBOX_STD_DEV
    # 转置掩码并增加一个维度
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # 获取每个正样本的最大重叠度对应的掩码
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
    # 获取正样本的提议框
    boxes = positive_rois
    # 如果使用小掩码,则将提议框的坐标转换为相对于真实框的坐标
    if config.USE_MINI_MASK:
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    # 获取每个掩码的索引
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    # 根据提议框的坐标在掩码上进行裁剪并调整大小
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # 去掉多余的维度
    masks = tf.squeeze(masks, axis=3)
    # 将掩码的值四舍五入为0或1
    masks = tf.round(masks)
    # 将正样本和负样本的提议框合并
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    # 获取负样本的数量
    N = tf.shape(negative_rois)[0]
    # 计算需要填充的数量
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    # 填充提议框
    rois = tf.pad(rois, [(0, P), (0, 0)])
    # 填充真实框
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    # 填充类别ID
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    # 填充偏移量
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    # 填充掩码
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])
    # 返回提议框、类别ID、偏移量和掩码
    return rois, roi_gt_class_ids, deltas, masks


def trim_zeros_graph(boxes, name='trim_zeros'):
    """
    如果前一步没有满POST_NMS_ROIS_TRAINING个建议框,会有padding
    要去掉padding
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


# 这是一个类,用于找到建议框的ground_truth。
# 它的call方法接收输入,包括建议框、每个真实框对应的类、真实框的位置和真实框的语义分割情况,
# 然后根据这些输入计算出
# 内部真实存在目标的建议框、每个建议框对应的类、每个建议框应该有的调整参数和每个建议框的语义分割情况。
class DetectionTargetLayer(Layer):
    # 初始化函数,接收一个配置对象和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化函数
        super(DetectionTargetLayer, self).__init__(**kwargs)
        # 保存配置对象
        self.config = config

    # call方法,接收输入并进行处理
    def call(self, inputs):
        # 提取输入中的提议框
        proposals = inputs[0]
        # 提取输入中的真实框类别ID
        gt_class_ids = inputs[1]
        # 提取输入中的真实框
        gt_boxes = inputs[2]
        # 提取输入中的真实框掩码
        gt_masks = inputs[3]

        # 对真实框进行编码
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        # 使用批处理切片函数处理输入,得到输出
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        # 返回处理后的输出
        return outputs

    # 计算输出形状的方法
    def compute_output_shape(self, input_shape):
        # 返回输出的形状
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    # 计算掩码的方法
    def compute_mask(self, inputs, mask=None):
        # 返回掩码,这里没有掩码,所以返回None
        return [None, None, None, None]
