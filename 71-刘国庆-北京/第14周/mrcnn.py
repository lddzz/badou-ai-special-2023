# mrcnn.py文件是实现Mask R-CNN模型的主要代码文件。Mask R-CNN是一种用于目标检测和实例分割的深度学习模型。
# 在这个文件中，定义了构建Mask R-CNN模型的各个部分，包括：
# 1. rpn_graph函数：用于构建RPN（Region Proposal Network）网络，生成候选区域。
# 2. build_rpn_model函数：用于建立RPN模型。
# 3. fpn_classifier_graph函数：用于构建FPN（Feature Pyramid Network）分类器，对RPN生成的候选区域进行分类和边界框回归。
# 4. build_fpn_mask_graph函数：用于构建FPN掩码图，对每个候选区域生成一个二值掩码。
# 5. get_predict_model函数：用于构建预测用的Mask R-CNN模型。
# 6. get_train_model函数：用于构建训练用的Mask R-CNN模型。
# 这个文件的主要作用是定义和构建Mask R-CNN模型，包括模型的各个组成部分以及训练和预测用的模型。
# 导入所需的库和模块
from keras.layers import Input, ZeroPadding2D, Conv2D, MaxPooling2D, BatchNormalization, Activation, UpSampling2D, Add, \
    Lambda, Concatenate
from keras.layers import Reshape, TimeDistributed, Dense, Conv2DTranspose
from keras.models import Model
import keras.backend as K
from nets.resnet import get_resnet  # 导入自定义的ResNet获取函数
from nets.layers import ProposalLayer, PyramidROIAlign, DetectionLayer, DetectionTargetLayer  # 导入自定义的各种层
from nets.mrcnn_training import *  # 导入Mask R-CNN训练模块中的所有内容
from utils.anchors import get_anchors  # 导入获取锚框的函数
from utils.utils import norm_boxes_graph, parse_image_meta_graph  # 导入一些工具函数
import tensorflow as tf
import numpy as np

'''
TimeDistributed:
对FPN网络输出的多层卷积特征进行共享参数。
TimeDistributed的意义在于使不同层的特征图共享权重。
'''


# ------------------------------------#
#   五个不同大小的特征层会传入到
#   RPN当中，获得建议框
# ------------------------------------#
def rpn_graph(feature_map, anchors_per_location):
    # 使用3x3的卷积核进行特征图的卷积，通道数为512，激活函数为ReLU
    shared = Conv2D(512, (3, 3), padding='same', activation='relu', name='rpn_conv_shared')(feature_map)
    # 利用1x1的卷积核进行类别的原始预测，通道数为2倍的先验框数量
    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')(shared)
    # 将预测结果的形状调整为[-1, 2]，代表每个先验框对应的两类别的预测得分
    rpn_class_logits = Reshape([-1, 2])(x)
    # 使用softmax激活函数获得先验框对应两个类别的概率
    rpn_probs = Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)
    # 利用1x1的卷积核进行先验框的位置调整，通道数为4倍的先验框数量
    x = Conv2D(anchors_per_location * 4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')(shared)
    # 将调整参数的形状调整为[-1, 4]，代表每个先验框的位置调整参数
    rpn_bbox = Reshape([-1, 4])(x)
    # 返回RPN网络的预测结果：类别预测得分、类别概率、位置调整参数
    return [rpn_class_logits, rpn_probs, rpn_bbox]


# ------------------------------------#
#   建立建议框网络模型
#   RPN模型
# ------------------------------------#
def build_rpn_model(anchors_per_location, depth):
    # 输入RPN网络的特征图
    input_feature_map = Input(shape=[None, None, depth], name="input_rpn_feature_map")
    # 调用rpn_graph构建RPN模型的输出
    outputs = rpn_graph(input_feature_map, anchors_per_location)
    # 构建RPN模型，输入为特征图，输出为类别原始预测、类别概率、位置调整预测
    return Model([input_feature_map], outputs, name="rpn_model")


# ------------------------------------#
#   建立classifier模型
#   这个模型的预测结果会调整建议框
#   获得最终的预测框
# ------------------------------------#
# 定义FPN分类器图的函数，接受感兴趣区域（rois）、特征图（feature_maps）、图像元信息（image_meta）等参数
def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024):
    # 使用金字塔ROI对齐层对感兴趣区域进行对齐，输入包括rois、image_meta和feature_maps
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # 对金字塔ROI对齐层的输出进行时间分布卷积，卷积核大小为(pool_size, pool_size)，有效填充
    x = TimeDistributed(Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"), name="mrcnn_class_conv1")(x)
    # 对卷积层的输出进行时间分布批量归一化，可选择是否在训练时进行归一化
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    # 对批量归一化后的输出进行ReLU激活
    x = Activation('relu')(x)
    # 再次对激活后的输出进行时间分布卷积，卷积核大小为(1, 1)
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
    # 对第二个卷积层的输出进行时间分布批量归一化
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    # 对第二个批量归一化后的输出进行ReLU激活
    x = Activation('relu')(x)
    # 使用Lambda层对第二个激活后的输出进行维度压缩，通过squeeze函数从第3和第2维度进行挤压
    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)
    # 对维度压缩后的输出进行时间分布全连接层，生成用于分类的logits
    mrcnn_class_logits = TimeDistributed(Dense(num_classes), name='mrcnn_class_logits')(shared)
    # 对分类logits进行时间分布softmax激活，生成分类概率
    mrcnn_probs = TimeDistributed(Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)
    # 对维度压缩后的输出进行时间分布全连接层，生成用于回归的bbox（边界框）
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)
    # 对回归bbox的输出进行形状重塑，生成最终的边界框
    mrcnn_bbox = Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
    # 返回分类logits、分类概率和回归bbox作为函数的输出
    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


# 定义FPN掩码图的构建函数，接受感兴趣区域（rois）、特征图（feature_maps）、图像元信息（image_meta）等参数
def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    # 使用金字塔ROI对齐层对感兴趣区域进行对齐，输入包括rois、image_meta和feature_maps
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)
    # 对金字塔ROI对齐层的输出进行时间分布卷积，卷积核大小为(3, 3)，使用相同填充
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    # 对卷积层的输出进行时间分布批量归一化，可选择是否在训练时进行归一化
    x = TimeDistributed(BatchNormalization(), name='mrcnn_mask_bn1')(x, training=train_bn)
    # 对卷积层输出进行ReLU激活
    x = Activation('relu')(x)
    # 重复上述过程，总共进行了4次卷积
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = Activation('relu')(x)
    # 反卷积，使用2x2的卷积核进行上采样
    x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    # 再次进行1x1卷积，调整通道数量，最终通道数为num_classes，代表分的类别
    x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)
    # 返回构建的掩码图
    return x


# 定义函数，用于构建预测用的Mask R-CNN模型
def get_predict_model(config):
    # 从配置对象中获取图像的高度和宽度
    h, w = config.IMAGE_SHAPE[:2]
    # 检查图像的高度和宽度是否可以被2的6次方整除，以确保在下采样和上采样时不会产生小数。否则，引发异常。
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
    # 创建用于输入图像的Keras Input层
    input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    # 创建包含必要信息的图像元数据的Keras Input层
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
    # 创建输入先验框的Keras Input层
    input_anchors = Input(shape=[None, 4], name="input_anchors")
    # 获取Resnet里的压缩程度不同的一些层
    _, C2, C3, C4, C5 = get_resnet(input_image, stage5=True, train_bn=config.TRAIN_BN)
    # 组合成特征金字塔的结构
    # P5长宽共压缩了5次，Height/32,Width/32,256
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    # P4长宽共压缩了4次，Height/16,Width/16,256
    P4 = Add(name="fpn_p4add")([
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    # P4长宽共压缩了3次，Height/8,Width/8,256
    P3 = Add(name="fpn_p3add")([
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    # P4长宽共压缩了2次，Height/4,Width/4,256
    P2 = Add(name="fpn_p2add")([
        UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
    # 各自进行一次256通道的卷积，此时P2、P3、P4、P5通道数相同
    # Height/4,Width/4,256
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    # Height/8,Width/8,256
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    # Height/16,Width/16,256
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    # Height/32,Width/32,256
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # 在建议框网络里面还有一个P6用于获取建议框
    # Height/64,Width/64,256
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    # P2, P3, P4, P5, P6可以用于获取建议框
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    # P2, P3, P4, P5用于获取mask信息
    mrcnn_feature_maps = [P2, P3, P4, P5]
    # 获取输入的先验框
    anchors = input_anchors
    # 构建RPN模型
    rpn = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
    # 初始化用于存储RPN网络输出的列表
    rpn_class_logits, rpn_class, rpn_bbox = [], [], []
    # 获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
    # 遍历RPN特征图列表，对每个特征图进行RPN网络的前向传播
    for p in rpn_feature_maps:
        logits, classes, bbox = rpn([p])
        # 将当前特征图的RPN网络输出的logits、classes和bbox存储到相应的列表中
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)
    # 将各个特征图的RPN网络输出在第1个维度上连接，形成一个大的输出
    rpn_class_logits = Concatenate(axis=1, name="rpn_class_logits")(rpn_class_logits)
    rpn_class = Concatenate(axis=1, name="rpn_class")(rpn_class)
    rpn_bbox = Concatenate(axis=1, name="rpn_bbox")(rpn_bbox)
    # 获取配置中指定的推理时的RoIs数量
    proposal_count = config.POST_NMS_ROIS_INFERENCE
    # Batch_size, proposal_count, 4
    # 对先验框进行解码
    # 使用ProposalLayer生成RoIs（Region of Interest）
    # 参数proposal_count：推理时生成的RoIs数量
    # 参数nms_threshold：非极大值抑制的阈值
    # 参数name：ProposalLayer的名称
    # 参数config：模型配置对象，包含了一些模型的超参数和设置
    # 输入是RPN网络的类别概率（rpn_class）、边界框偏移（rpn_bbox）以及输入的先验框（anchors）
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])
    # 获得classifier的结果
    # 调用fpn_classifier_graph函数，构建FPN分类器部分的计算图
    # 输出包括分类概率的logits（mrcnn_class_logits）、类别概率分布（mrcnn_class）以及边界框回归的结果（mrcnn_bbox）
    # 参数rpn_rois：RoIs（Region of Interest），是由RPN网络生成的建议框
    # 参数mrcnn_feature_maps：用于提取特征的FPN特征图列表
    # 参数input_image_meta：输入图像的元数据
    # 参数config.POOL_SIZE：Pooling层的大小
    # 参数config.NUM_CLASSES：类别的数量
    # 参数train_bn：是否在训练时更新Batch Normalization层的权重
    # 参数config.FPN_CLASSIF_FC_LAYERS_SIZE：FPN分类器的全连接层的大小
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
        fpn_classifier_graph(rpn_rois,
                             mrcnn_feature_maps,
                             input_image_meta,
                             config.POOL_SIZE,
                             config.NUM_CLASSES,
                             train_bn=config.TRAIN_BN,
                             fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
    # 生成检测结果
    # 使用DetectionLayer生成检测结果
    # 参数config：模型配置对象，包含了一些模型的超参数和设置
    # 参数name：DetectionLayer的名称
    # 输入包括RoIs（rpn_rois）、分类概率分布（mrcnn_class）、边界框回归的结果（mrcnn_bbox）以及输入图像的元数据（input_image_meta）
    detections = DetectionLayer(config, name="mrcnn_detection")(
        [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
    # 使用Lambda层提取检测结果中的边界框信息
    detection_boxes = Lambda(lambda x: x[..., :4])(detections)
    # 调用函数生成Mask R-CNN的Mask部分的输出
    # 调用build_fpn_mask_graph函数，构建FPN Mask部分的计算图
    # 输出包括Mask预测的结果（mrcnn_mask）
    # 参数detection_boxes：检测结果中的边界框信息
    # 参数mrcnn_feature_maps：用于提取特征的FPN特征图列表
    # 参数input_image_meta：输入图像的元数据
    # 参数config.MASK_POOL_SIZE：Mask部分的Pooling层的大小
    # 参数config.NUM_CLASSES：类别的数量
    # 参数train_bn：是否在训练时更新Batch Normalization层的权重
    mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                      input_image_meta,
                                      config.MASK_POOL_SIZE,
                                      config.NUM_CLASSES,
                                      train_bn=config.TRAIN_BN)
    # 定义整个模型，包括输入和输出 定义整个Mask R-CNN模型，包括输入和输出 参数[input_image, input_image_meta, input_anchors]：模型的输入，分别是图像、图像元数据和先验框
    # 参数[detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class,
    # rpn_bbox]：模型的输出，包括检测结果、分类概率、边界框回归、Mask预测、RoIs以及RPN网络的输出 参数name='mask_rcnn'：模型的名称
    model = Model([input_image, input_image_meta, input_anchors],
                  [detections, mrcnn_class, mrcnn_bbox,
                   mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                  name='mask_rcnn')
    # 返回构建的Mask R-CNN模型
    return model


# 定义获取训练模型的函数，接收一个配置对象作为参数
def get_train_model(config):
    # 获取图像的高度和宽度
    h, w = config.IMAGE_SHAPE[:2]
    # 检查图像大小是否满足要求，要求是高度和宽度都能被 2 整除至少 6 次
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        # 如果不满足要求，抛出异常
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
    # 定义输入图像的张量，形状为 [None, None, 图像通道数]
    input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    # 定义输入图像的元信息张量，形状为 [图像元信息大小]
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
    # 定义输入 RPN（Region Proposal Network）匹配信息的张量，形状为 [None, 1]
    input_rpn_match = Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
    # 定义输入 RPN 边界框信息的张量，形状为 [None, 4]
    input_rpn_bbox = Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
    # 定义输入真实框类别信息的张量，形状为 [None]
    input_gt_class_ids = Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
    # 定义输入真实框位置信息的张量，形状为 [None, 4]
    input_gt_boxes = Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
    # 使用 Lambda 函数将真实框位置信息标准化到 0-1 之间，根据输入图像的形状进行标准化
    gt_boxes = Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)
    # 根据配置决定是否使用迷你掩码（Mini Mask）
    if config.USE_MINI_MASK:
        # 如果使用迷你掩码，定义输入迷你掩码的张量，形状为 [迷你掩码高度, 迷你掩码宽度, None]
        input_gt_masks = Input(shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                               name="input_gt_masks", dtype=bool)
    else:
        # 如果不使用迷你掩码，定义输入掩码的张量，形状为 [图像高度, 图像宽度, None]
        input_gt_masks = Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None], name="input_gt_masks",
                               dtype=bool)
    # 使用 ResNet 网络获取不同压缩程度的特征层
    _, C2, C3, C4, C5 = get_resnet(input_image, stage5=True, train_bn=config.TRAIN_BN)
    # 使用 1x1 卷积层构建特征金字塔的顶层 P5
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    # 使用上采样和 1x1 卷积层构建特征金字塔中的 P4
    P4 = Add(name="fpn_p4add")([
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    # 使用上采样和 1x1 卷积层构建特征金字塔中的 P3
    P3 = Add(name="fpn_p3add")([
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    # 使用上采样和 1x1 卷积层构建特征金字塔中的 P2
    P2 = Add(name="fpn_p2add")([
        UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
    # 各自进行一次256通道的卷积，此时P2、P3、P4、P5通道数相同
    # 对特征金字塔的每一层进行 3x3 卷积，确保形状一致
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # 使用最大池化进行下采样，得到特征金字塔的底层 P6
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    # 定义用于 RPN（Region Proposal Network）的特征图列表
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    # 定义用于 Mask R-CNN 的特征图列表
    mrcnn_feature_maps = [P2, P3, P4, P5]
    # 获取锚框（anchors）的信息
    anchors = get_anchors(config, config.IMAGE_SHAPE)
    # 将锚框的形状进行广播，扩展为与批次大小相匹配
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
    # 将锚框转化为 TensorFlow 变量，并命名为 "anchors"
    anchors = Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
    # 构建 RPN（Region Proposal Network）模型
    rpn = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
    # 初始化用于存储 RPN 预测结果的列表
    rpn_class_logits, rpn_class, rpn_bbox = [], [], []
    # 获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
    # 遍历 RPN 特征图列表，获取每个特征图对应的 RPN 预测结果
    for p in rpn_feature_maps:
        logits, classes, bbox = rpn([p])
        # 将 RPN 预测结果添加到对应的列表中
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)
    # 在类别维度上拼接每个特征图的RPN类别预测得分，形成完整的RPN类别预测结果
    rpn_class_logits = Concatenate(axis=1, name="rpn_class_logits")(rpn_class_logits)
    # 在类别维度上拼接每个特征图的RPN类别预测结果，形成完整的RPN类别预测结果
    rpn_class = Concatenate(axis=1, name="rpn_class")(rpn_class)
    # 在边界框维度上拼接每个特征图的RPN边界框预测结果，形成完整的RPN边界框预测结果
    rpn_bbox = Concatenate(axis=1, name="rpn_bbox")(rpn_bbox)
    # 设置用于训练的候选建议框的数量，即在非极大值抑制（NMS）之后保留的建议框数量
    proposal_count = config.POST_NMS_ROIS_TRAINING
    # 使用 ProposalLayer 层生成 RPN 预测框
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,  # 设置候选建议框的数量，即在非极大值抑制（NMS）之后保留的建议框数量
        nms_threshold=config.RPN_NMS_THRESHOLD,  # NMS 阈值，用于过滤高度重叠的建议框
        name="ROI",  # 层的名称
        config=config  # 配置对象，包含模型的参数和设置
    )([rpn_class, rpn_bbox, anchors])  # 输入是 RPN 的类别预测、边界框预测和锚框
    # 使用 Lambda 函数从输入图像元信息中提取激活的类别标识
    active_class_ids = Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)
    # 如果不使用 RPN 预测框（而是使用外部输入的建议框）
    if not config.USE_RPN_ROIS:
        # 定义输入建议框的张量，形状为 [训练时保留的建议框数量, 4]
        input_rois = Input(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32)
        # 使用 Lambda 函数将输入建议框标准化到 0-1 之间，根据输入图像的形状进行标准化
        target_rois = Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_rois)
    else:
        # 如果使用 RPN 预测框，则直接使用 RPN 预测的建议框
        target_rois = rpn_rois
    # 通过 DetectionTargetLayer 层获取建议框的 ground_truth 信息
    rois, target_class_ids, target_bbox, target_mask = \
        DetectionTargetLayer(config, name="proposal_targets")([
            target_rois,  # 输入的建议框
            input_gt_class_ids,  # 真实框的类别标识
            gt_boxes,  # 真实框的位置信息
            input_gt_masks  # 真实框的语义分割信息
        ])
    # 利用 FPN 分类器图（fpn_classifier_graph）获取建议框的分类预测结果
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
        fpn_classifier_graph(
            rois,  # 建议框
            mrcnn_feature_maps,  # Mask R-CNN 使用的特征图列表
            input_image_meta,  # 输入图像的元信息
            config.POOL_SIZE,  # 池化层的大小
            config.NUM_CLASSES,  # 类别数量
            train_bn=config.TRAIN_BN,  # 是否在训练时更新 Batch Normalization 层
            fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE  # FPN 分类器的全连接层大小
        )
    # 利用 FPN 掩码图（build_fpn_mask_graph）获取建议框的掩码预测结果
    mrcnn_mask = build_fpn_mask_graph(
        rois,  # 建议框
        mrcnn_feature_maps,  # Mask R-CNN 使用的特征图列表
        input_image_meta,  # 输入图像的元信息
        config.MASK_POOL_SIZE,  # 掩码池化层的大小
        config.NUM_CLASSES,  # 类别数量
        train_bn=config.TRAIN_BN  # 是否在训练时更新 Batch Normalization 层
    )
    # 定义输出建议框的张量，与输入建议框保持一致
    output_rois = Lambda(lambda x: x * 1, name="output_rois")(rois)
    # 使用 Lambda 函数计算 RPN 模型的分类损失
    rpn_class_loss = Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
        [input_rpn_match, rpn_class_logits])
    # 使用 Lambda 函数计算 RPN 模型的边界框回归损失
    rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
        [input_rpn_bbox, input_rpn_match, rpn_bbox])
    # 使用 Lambda 函数计算 Mask R-CNN 模型的分类损失
    class_loss = Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
        [target_class_ids, mrcnn_class_logits, active_class_ids])
    # 使用 Lambda 函数计算 Mask R-CNN 模型的边界框回归损失
    bbox_loss = Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
        [target_bbox, target_class_ids, mrcnn_bbox])
    # 使用 Lambda 函数计算 Mask R-CNN 模型的掩码损失
    mask_loss = Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
        [target_mask, target_class_ids, mrcnn_mask])
    # 定义模型的输入层，包括图像、元信息、RPN 匹配、RPN 边界框、真实框类别、真实框位置和真实框掩码
    inputs = [
        input_image,  # 输入图像
        input_image_meta,  # 输入图像的元信息
        input_rpn_match,  # RPN 匹配信息
        input_rpn_bbox,  # RPN 边界框信息
        input_gt_class_ids,  # 真实框的类别信息
        input_gt_boxes,  # 真实框的位置信息
        input_gt_masks  # 真实框的掩码信息
    ]
    # 如果不使用 RPN 生成的建议框，则添加外部输入的建议框信息
    if not config.USE_RPN_ROIS:
        inputs.append(input_rois)
    # 定义模型的输出层，包括 RPN 模型和 Mask R-CNN 模型的各项输出
    outputs = [
        rpn_class_logits,  # RPN 模型的分类预测得分
        rpn_class,  # RPN 模型的分类预测类别
        rpn_bbox,  # RPN 模型的边界框预测结果
        mrcnn_class_logits,  # Mask R-CNN 模型的分类预测得分
        mrcnn_class,  # Mask R-CNN 模型的分类预测类别
        mrcnn_bbox,  # Mask R-CNN 模型的边界框预测结果
        mrcnn_mask,  # Mask R-CNN 模型的掩码预测结果
        rpn_rois,  # RPN 模型生成的建议框
        output_rois,  # 模型输出的建议框
        rpn_class_loss,  # RPN 模型的分类损失
        rpn_bbox_loss,  # RPN 模型的边界框回归损失
        class_loss,  # Mask R-CNN 模型的分类损失
        bbox_loss,  # Mask R-CNN 模型的边界框回归损失
        mask_loss  # Mask R-CNN 模型的掩码损失
    ]
    # 创建模型对象，将输入层和输出层传入，命名为 'mask_rcnn'
    model = Model(inputs, outputs, name='mask_rcnn')
    # 返回构建好的模型
    return model
