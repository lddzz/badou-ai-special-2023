# 导入ResNet50类和classifier_layers函数,它们位于nets.resnet模块中
from nets.resnet import ResNet50, classifier_layers
# 导入Keras中的各种层,包括卷积层、输入层、时间分布层、展平层、全连接层以及重塑层
from keras.layers import Conv2D, Input, TimeDistributed, Flatten, Dense, Reshape
# 导入Keras中的Model类,用于构建神经网络模型
from keras.models import Model
# 导入RoiPoolingConv类,该类用于实现RoiPoolingConv层的功能,位于nets.RoiPoolingConv模块中
from nets.RoiPoolingConv import RoiPoolingConv


# 定义获取区域提议网络(RPN)的函数get_rpn,
# 接收基础层base_layers和锚点数量num_anchors作为输入参数
def get_rpn(base_layers, num_anchors):
    # 使用3x3的卷积核对基础层(特征图)进行卷积,输出512个通道,
    # 激活函数为ReLU,采用"same"填充以保持特征图大小不变,卷积核初始化方式normal,命名为rpn_conv1
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)
    # 添加1x1的卷积层来预测每个锚点的目标类别,输出通道数与锚点数量相同,
    # 使用Sigmoid激活函数进行二分类(目标/非目标),卷积核初始化方式uniform,命名为rpn_out_class
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    # 添加另一个1x1的卷积层来预测边界框的调整值,输出通道数是锚点数量的四倍(每个锚点预测四个坐标值),
    # 使用线性激活函数,卷积核初始化方式zero,命名为rpn_out_regress
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    # 重塑类别预测层的输出,使其成为二维形式,其中第二维是1,表示每个锚点的分类预测,命名为classification
    x_class = Reshape((-1, 1), name="classification")(x_class)
    # 重塑边界框回归层的输出,使其成为二维形式,其中第二维是4,表示每个锚点的边界框坐标预测,命名为regression
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    # 返回一个列表,包含处理后的类别预测值x_class、边界框回归值x_regr和传入的基础层base_layers
    return [x_class, x_regr, base_layers]


# 定义获取分类器的函数get_classifier
# 基础层base_layers、感兴趣区域的输入input_rois、感兴趣区域的数量num_rois、
# 类别数nb_classes=21、是否可训练标志trainable=False
def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    # 定义池化区域pooling_regions的大小14,这是RoiPooling层处理后每个ROI的固定尺寸
    pooling_regions = 14
    # 定义传递给分类器的输入数据形状,包括ROI数量、池化区域的尺寸和特征图的通道数14, 14, 1024
    input_shape = (num_rois, 14, 14, 1024)
    # 使用RoiPoolingConv层对感兴趣区域进行池化操作,对输入的基础层和ROI进行操作,产生固定尺寸(pooling_regions, num_rois)的池化特征图
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    # 使用可能自定义的classifier_layers函数对池化后的特征图进行进一步处理,trainable参数控制这些层的可训练性
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    # 使用TimeDistributed包装器和Flatten层将特征图展平,以便应用全连接层
    out = TimeDistributed(Flatten())(out)
    # 添加全连接层进行分类,使用TimeDistributed包装器,对每个ROI输出类别预测,使用softmax激活函数,
    # 卷积核初始化方式zero,命名为f"dense_class_{nb_classes}"
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # 添加全连接层进行边界框回归,使用TimeDistributed包装器,对每个ROI输出边界框调整值,使用线性激活函数,
    # 卷积核初始化方式zero,命名为f"dense_regress_{nb_classes}"
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    # 函数返回包含类别预测值和边界框回归预测值的列表
    return [out_class, out_regr]


# 定义一个函数get_model用于构建整个目标检测模型,config配置信息、num_classes类别总数
def get_model(config, num_classes):
    # 输入层：定义模型的输入,接受任意尺寸的RGB图像(3个通道)
    inputs = Input(shape=(None, None, 3))
    # 感兴趣区域(ROI)的输入层：用于接收ROI的坐标,每个ROI由4个数值(坐标)表示
    roi_input = Input(shape=(None, 4))
    # 基础层：使用ResNet50作为特征提取网络,提取图像的特征图
    base_layers = ResNet50(inputs)
    # 计算锚框数量num_anchors：根据配置中的锚框尺寸* 比例,计算总共需要的锚框数量
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    # RPN层：调用get_rpn函数,创建RPN,基础层base_layers和锚点数量num_anchors,它负责生成区域提议
    rpn = get_rpn(base_layers, num_anchors)
    # RPN模型：定义一个模型,以图像作为输入,输出RPN的类别预测和边界框回归预测
    model_rpn = Model(inputs, rpn[:2])
    # 分类器层：调用get_classifier函数,创建分类器,它负责对RPN提出的区域进行分类和边界框回归
    # 创建分类器网络
    # base_layers：用于提取特征的基础网络输出,roi_input：感兴趣区域的坐标输入,
    # config.num_rois：同时处理的ROI数量,nb_classes：分类任务中的类别总数,trainable：是否在训练过程中更新权重
    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    # 分类器模型：定义一个模型,以图像和ROI作为输入,输出分类结果和边界框回归结果
    model_classifier = Model([inputs, roi_input], classifier)
    # 整体模型：定义一个模型,整合了RPN和分类器的功能,以图像和ROI作为输入,同时输出RPN和分类器的结果
    model_all = Model([inputs, roi_input], rpn[:2] + classifier)
    # 返回RPN模型、分类器模型和整体模型
    return model_rpn, model_classifier, model_all


# 定义一个函数get_predict_model,用于构建用于预测的目标检测模型,config配置信息的对象,num_classes类别数量作为参数,
def get_predict_model(config, num_classes):
    # 输入层：定义模型的输入,接受任意尺寸的RGB图像(3个通道)
    inputs = Input(shape=(None, None, 3))
    # 感兴趣区域(ROI)的输入层：用于接收ROI的坐标,每个ROI由4个数值(坐标)表示
    roi_input = Input(shape=(None, 4))
    # 特征图的输入层：接收预先计算好的特征图,形状为(None, None, 1024)
    feature_map_input = Input(shape=(None, None, 1024))
    # 使用ResNet50作为基础网络：从输入图像中提取特征图
    base_layers = ResNet50(inputs)
    # 计算锚框数量：根据配置中的锚框尺度和比例
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    # 获取RPN的输出：调用get_rpn函数,创建RPN,输入是基础层base_layers和锚点数量num_anchors,它负责生成区域提议
    rpn = get_rpn(base_layers, num_anchors)
    # 创建RPN模型：定义一个模型,输入是模型输入,输出RPN的类别和边界框预测
    model_rpn = Model(inputs, rpn)
    # 获取分类器的输出：调用get_classifier函数,创建分类器,它负责对ROI进行分类和边界框回归
    # base_layers：用于提取特征的基础网络输出,roi_input：感兴趣区域的坐标输入,
    # config.num_rois：同时处理的ROI数量,nb_classes：分类任务中的类别总数,trainable：是否在训练过程中更新权重
    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    # 创建只包含分类器的模型：定义一个模型,输入是特征图和感兴趣区域，输出是分类器的类别预测值和边界框回归预测值
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    # 返回RPN模型和只包含分类器的模型
    return model_rpn, model_classifier_only
