# 导入进行数值计算和数组操作的库NumPy
import numpy as np
# 提供颜色空间转换的函数
import colorsys
# 用于与操作系统交互的库
import os
# 导入自定义的Faster R-CNN模块
import nets.frcnn as frcnn
# 导入Faster R-CNN模型训练相关模块
from nets.frcnn_training import get_new_img_size
# 后端Keras，用于访问深度学习框架的底层功能
from keras import backend as K
# 用于图像预处理的函数，通常用于将图像数据转换成模型可以处理的格式
from keras.applications.imagenet_utils import preprocess_input
# Python Imaging Library(PIL)模块，用于图像处理和绘图
from PIL import Image, ImageFont, ImageDraw
# 用于处理边界框的实用工具类
from utils.utils import BBoxUtility
# 用于获取锚框的函数
from utils.anchors import get_anchors
# Faster R-CNN模型的配置类
from utils.config import Config
# Python的拷贝模块，用于对象的复制
import copy
# Python的数学模块，提供了一些基本的数学函数
import math


# 定义 Faster R-CNN 类
class FRCNN(object):
    # 默认配置参数
    _defaults = {
        "model_path": 'model_data/voc_weights.h5',  # 模型权重文件路径
        "classes_path": 'model_data/voc_classes.txt',  # 分类类别文件路径
        "confidence": 0.7,  # 置信度阈值
    }

    # 类方法装饰器，用于定义类方法
    @classmethod
    # 定义一个名为 get_defaults 的类方法，接收两个参数：cls 和 n
    # 获取默认配置参数
    def get_defaults(cls, n):
        # 检查 n 是否为 cls._defaults 字典的一个键
        if n in cls._defaults:
            # 如果是，返回这个键对应的值
            return cls._defaults[n]
        # 如果属性名称未在默认配置中找到，返回未识别的属性名信息
        else:
            # 返回一个表示 n 是未识别属性名的字符串
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化faster RCNN
    # ---------------------------------------------------#
    def __init__(self):
        # 使用默认配置参数初始化实例属性
        self.__dict__.update(self._defaults)
        # 获取分类类别列表
        self.class_names = self._get_class()
        # 获取当前会话(session)对象
        self.sess = K.get_session()
        # 创建配置对象
        self.config = Config()
        # 调用生成方法，构建模型及相关配置
        self.generate()
        # 创建边界框实用工具对象
        self.bbox_util = BBoxUtility()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    # 定义_get_class方法，用于从文件中获取分类的类别
    def _get_class(self):
        # 将类别文件的相对路径转换为绝对路径
        class_path = os.path.expanduser(self.classes_path)
        # 打开类别文件进行读取
        with open(class_path) as f:
            # 读取文件中的每一行，每行代表一个类别名称
            class_names = f.readlines()
        # 移除每个类别名称前后的空白字符
        class_names = [c.strip() for c in class_names]
        # 返回处理后的类别名称列表
        return class_names

    # ---------------------------------------------------#
    #   加载和准备目标检测模型
    # ---------------------------------------------------#
    def generate(self):
        # 将模型路径扩展为绝对路径
        model_path = os.path.expanduser(self.model_path)
        # 确保模型路径以 '.h5' 结尾，否则抛出异常
        assert model_path.endswith('.h5'), "Keras 模型权重路径必须以'.h5'结尾"
        # 计算总的种类数量,设置 self.num_classes 的值，这个值是类别名称列表长度加 1。
        # 加 1 是因为通常在物体检测中需要考虑一个额外的“背景”类别，所以总类别数比实际类别列表的长度多一个。
        self.num_classes = len(self.class_names) + 1

        # 输入config和总的种类数量,获取预测模型，包括RPN和分类器
        self.model_rpn, self.model_classifier = frcnn.get_predict_model(self.config, self.num_classes)
        # 载入 RPN 模型的权重,参数 'by_name=True' 表示按名称加载权重
        self.model_rpn.load_weights(self.model_path, by_name=True)
        # 载入分类器模型的权重,
        # by_name=True表示按名称加载权重
        # skip_mismatch=True表示如果权重文件中的层与模型中的层不匹配，则跳过这些层。
        self.model_classifier.load_weights(self.model_path, by_name=True, skip_mismatch=True)
        # 打印模型、锚框和类别信息
        print(f"{model_path} 打印模型、锚框和类别信息已经加载")

        # 创建一个 HSV 颜色元组的列表 hsv_tuples。
        # 列表中的每个元组代表一个颜色，用 HSV 色彩空间表示。
        # x / len(self.class_names) 计算每个类别对应的色调(Hue)值，其范围在 0 到 1 之间。
        # 通过 for x in range(len(self.class_names)) 遍历所有类别。
        # 每个元组中的 1. 表示饱和度(Saturation)和明度(Value)都设置为最大值 1。
        # 这样做的目的是为每个类别生成一个明亮鲜艳的颜色。
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]

        # 将HSV颜色元组列表转换为RGB颜色
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # 将RGB颜色值的元组列表进行缩放，确保每个通道值在0-255范围内
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    # ---------------------------------------------------#
    #   获取经过卷积层处理后的图像输出长度
    # ---------------------------------------------------#
    # 定义一个函数get_img_output_length，用于计算经过卷积操作后的图像输出尺寸
    # 参数包括 self(表示对象实例，通常在类中使用)、width(图像的宽度)、height(图像的高度)
    def get_img_output_length(self, width, height):

        # 定义一个内部函数get_output_length,input_length:卷积层输入长度,用于计算经过卷积层后的输出长度
        def get_output_length(input_length):
            # 定义卷积核filter_sizes的大小，这里定义了四个卷积层的核大小[7, 3, 1, 1]
            filter_sizes = [7, 3, 1, 1]
            # 定义每个卷积层的填充padding大小[3, 1, 0, 0]
            padding = [3, 1, 0, 0]
            # 定义步幅stride，此处所有卷积层使用相同的步幅2
            stride = 2
            # 循环遍历每个卷积层
            for i in range(4):
                # 这个循环会遍历四次，对应四个卷积层。
                # input_length 表示当前卷积层的输入长度，这里计算的是下一层的输出长度。
                # 具体计算公式是：(输入长度 + 2 * 填充 - 卷积核大小) // 步幅 + 1
                input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1

            # 返回经过所有卷积层处理后的长度
            return input_length

        # 调用内部函数，分别计算并返回宽度和高度经过卷积处理后的长度
        return get_output_length(width), get_output_length(height)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    # 1.获取图像形状：从输入的图像中获取宽度和高度信息。
    # 2.图像预处理：对图像进行大小调整，并进行归一化处理。
    # 3.使用RPN模型进行预测：使用区域提议网络（RegionProposalNetwork, RPN）模型对图像进行处理，以识别可能包含目标的区域。
    # 4.处理RPN输出：对RPN的输出结果进行解码，获取候选区域的坐标。
    # 5.坐标转换和处理：将候选区域的相对坐标转换为原图中的绝对坐标，并进行一些处理，如删除无效的边框。
    # 6.使用分类器模型进行预测：对每个提议区域使用分类器模型进行预测，识别区域中的对象，并获取其边框和类别。
    # 7.调整边框位置：根据分类器的输出调整边框位置，以更准确地框选出目标。
    # 8.筛选和非极大值抑制（NMS）：对检测到的边框进行筛选，保留置信度高的边框，并通过NMS去除重叠边框。
    # 9.绘制边框和标签：在原图上绘制检测到的边框，并在每个边框旁边标注类别和置信度。
    # 10.返回最终图像：返回绘制了边框和标签的图像。

    def detect_image(self, image):
        # ---------------------------------------------------#
        # Step 1: 获取图像形状信息
        # ---------------------------------------------------#
        # 获取输入图像的形状信息
        # 使用 NumPy 获取图像的宽度和高度，并存储在名为 image_shape 的 NumPy 数组中
        image_shape = np.array(np.shape(image)[0:2])
        # 从 image_shape 数组中提取图像的宽度和高度
        old_width = image_shape[1]
        old_height = image_shape[0]
        # 创建图像的深拷贝，存储在 old_image 变量中,以便后续的操作不会修改原始图像
        old_image = copy.deepcopy(image)

        # ---------------------------------------------------#
        # Step 2: 图像预处理
        # ---------------------------------------------------#
        # 获取新的图像大小
        width, height = get_new_img_size(old_width, old_height)
        # 对输入图像进行resize
        image = image.resize([width, height])
        # 将图像转换为NumPy数组，并使用float64类型存储
        photo = np.array(image, dtype=np.float64)
        # 图片预处理，归一化preprocess_input
        # 使用 NumPy 的 expand_dims 函数，在图像数据 photo 的前面添加一个维度，
        # 将其从 (H, W, C) 变为 (1, H, W, C)
        photo = preprocess_input(np.expand_dims(photo, axis=0))

        # ---------------------------------------------------#
        # Step 3: 使用RPN模型进行预测
        # ---------------------------------------------------#
        # 使用RPN模型进行预测
        preds = self.model_rpn.predict(photo)

        # ---------------------------------------------------#
        # Step 4: 处理RPN输出
        # ---------------------------------------------------#
        # 将预测结果进行解码
        # get_anchors获取锚点信息
        anchors = get_anchors(self.get_img_output_length(width, height), width, height)
        # 使用边界框实用程序self.bbox_util.detection_out对RPN模型的预测进行解码。
        # 'preds' 包含了RPN模型的输出，'anchors' 是预定义的边界框集合。
        # '1' 表示处理一个图像的批量大小。
        # 'confidence_threshold=0' 设置了一个阈值，用于筛选RPN模型预测的提议区域。
        # 这个阈值设为0，意味着不进行筛选，保留所有预测的提议区域。
        rpn_result = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)
        # 从RPN模型解码后的结果中提取边界框坐标。
        # 'rpn_results' 是一个数组，其中每个元素包含了预测的边界框信息。
        # 'rpn_results[0]' 选择了第一个元素，通常是因为处理的是单个图像。
        # '[:, 2:]' 是一个切片操作，用于选择每个边界框的坐标信息。
        # 通常，边界框的数据格式为 [x_min, y_min, x_max, y_max]，表示边界框的左上角和右下角坐标。
        # 这里的切片操作跳过了前两个元素，只选取了边界框的坐标。
        R = rpn_result[0][:, 2:]

        # ---------------------------------------------------#
        # Step 5: 坐标转换和处理
        # ---------------------------------------------------#
        # 对解码后的位置信息进行一系列处理，将相对坐标转换为原图中的绝对坐标
        # 计算边框左上角 x 坐标，通过将相对坐标乘以原图宽度与RPN步长的比值
        R[:, 0] = np.array(np.round(R[:, 0] * width / self.config.rpn_stride), dtype=np.int32)
        # 计算边框左上角 y 坐标，通过将相对坐标乘以原图高度与RPN步长的比值
        R[:, 1] = np.array(np.round(R[:, 1] * height / self.config.rpn_stride), dtype=np.int32)
        # 计算边框宽度，通过将相对宽度乘以原图宽度与RPN步长的比值
        R[:, 2] = np.array(np.round(R[:, 2] * width / self.config.rpn_stride), dtype=np.int32)
        # 计算边框高度，通过将相对高度乘以原图高度与RPN步长的比值
        R[:, 3] = np.array(np.round(R[:, 3] * height / self.config.rpn_stride), dtype=np.int32)
        # 对处理后的边框信息进行进一步调整，计算边框的宽度和高度
        # 计算边框的宽度，通过减去左上角 x 坐标得到
        R[:, 2] -= R[:, 0]
        # 计算边框的高度，通过减去左上角 y 坐标得到
        R[:, 3] -= R[:, 1]
        # 获取RPN模型的第三个输出作为基础层
        base_layer = preds[2]
        # 删除边框信息中宽度或高度小于1的无效边框
        # 创建一个空列表来存储要删除的无效边框的索引
        delete_line = []
        # 遍历所有边框，判断它们的宽度或高度是否小于1
        for i, r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                # 如果宽度或高度小于1，则将该边框的索引添加到delete_line列表中
                delete_line.append(i)
        # 使用NumPy的delete函数删除无效边框，axis=0表示按行删除
        R = np.delete(R, delete_line, axis=0)

        # ---------------------------------------------------#
        # Step 6: 使用分类器模型进行预测
        # ---------------------------------------------------#
        # 初始化用于存储边框bboxes、概率probs和标签信息labels的空列表
        bboxes = []
        probs = []
        labels = []
        # 遍历边界框数组R中的元素，以分批次处理每个边界框。
        # R.shape[0] 表示边界框的总数。
        # self.config.num_rois 是每个批次要处理的边界框的最大数量。
        # 使用整数除法 '//' 确定需要多少个完整批次来处理所有边界框，
        # 然后 '+ 1' 确保即使最后一个批次不满也能被处理。
        # 这样可以确保所有边界框都会被遍历和处理，即使它们的数量不是 self.config.num_rois 的整数倍。
        for jk in range(R.shape[0] // self.config.num_rois + 1):
            # 将每次处理的ROIs取出，并在第0维度上扩展维度，存储到ROIs列表中
            ROIs = np.expand_dims(R[self.config.num_rois * jk:self.config.num_rois * (jk + 1), :], axis=0)
            # 如果当前批次的ROIs数量为0，跳出循环
            if ROIs.shape[1] == 0:
                break
            # 如果当前批次是最后一个批次，进行填充操作
            if jk == R.shape[0] // self.config.num_rois:
                # 记录当前ROIs的形状
                curr_shape = ROIs.shape
                # 设置目标形状 target_shape 用于填充最后一个边界框批次。
                # curr_shape[0] 表示当前批次的维度，通常是1，因为一次处理一个图像。
                # self.config.num_rois 是每个批次应该包含的边界框的数量。
                # curr_shape[2] 表示边界框的属性数量，通常是4（即 x, y, width, height）。
                # 这样，target_shape 就定义了一个形状，其中包含足够的空间来填充最后一个边界框批次，
                # 以确保它有与其他批次相同的数量的边界框。
                target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                # 创建一个全零的数组 'ROIs_padded',其形状由 'target_shape' 指定,数据类型与原始边界框数组 'ROIs' 的数据类型相同
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                # 将原始边界框 (ROIs) 的数据复制到填充后的边界框数组 (ROIs_padded) 中。
                # 'ROIs_padded[:, :curr_shape[1], :]' 指定了目标数组的一个子区域，
                # 其中 ':' 表示所有批次，':curr_shape[1]' 表示每个批次中实际存在的边界框的数量，
                # 最后一个 ':' 表示边界框的所有属性（如x, y, 宽度和高度）。
                # 'ROIs' 是原始的边界框数组，其数据被复制到 'ROIs_padded' 的指定子区域中。
                # 这样做是为了保留原始边界框的数据，同时确保每个批次具有统一的形状。
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                # 使用第0维度的第一个ROIs填充剩余位置
                # 使用当前批次的第一个边界框 'ROIs[0, 0, :]' 来填充 'ROIs_padded' 数组的剩余部分。
                # 'ROIs_padded[0, curr_shape[1]:, :]' 指定从 'ROIs_padded' 的第一个元素开始，
                # 从 'curr_shape[1]'（当前批次实际边界框数量）到结束的位置进行填充。
                # 这样做是为了确保即使在边界框数量不足以填满一个批次时，
                # 'ROIs_padded' 也能保持统一的形状，从而适合后续的批处理操作。
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                # 更新ROIs为填充后的ROIs_padded
                ROIs = ROIs_padded
            # 使用分类器模型进行预测，得到分类概率和回归信息
            [P_cls, P_regr] = self.model_classifier.predict([base_layer, ROIs])
            # 遍历分类器模型的输出，处理每个感兴趣区域
            for i in range(P_cls.shape[1]):
                # 判断当前类别的最大概率是否满足置信度要求，并且不是背景类别
                # 'np.max(P_cls[0, ii, :]) < self.confidence' 检查最大置信度是否低于预设阈值 'self.confidence'。
                #   如果是，则认为模型对于该边界框的分类不够自信，因此跳过当前边界框。
                # 'np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1)' 检查预测的最高置信度是否对应于最后一个类别。
                #   在很多对象检测模型中，最后一个类别通常是用于表示“背景”或“无对象”的类别。
                if np.max(P_cls[0, i, :]) < self.confidence or np.argmax(P_cls[0, i, :]) == (P_cls.shape[2] - 1):
                    continue
                # 获取当前类别的标签
                label = np.argmax(P_cls[0, i, :])
                # 获取当前感兴趣区域的坐标和大小x, y, w, h
                (x, y, w, h) = ROIs[0, i, :]
                # 获取当前类别在分类输出中的索引cls_num
                cls_num = np.argmax(P_cls[0, i, :])
                # 获取当前感兴趣区域的回归信息
                # 从分类器模型的回归输出中提取特定类别的回归参数。
                # 'P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]' 表示选择当前边界框对应的回归参数。
                # '0' 表示选择第一个元素（通常是因为处理单个图像）。
                # 'ii' 表示当前正在处理的边界框的索引。
                # '4 * cls_num' 到 '4 * (cls_num + 1)' 表示提取与当前预测类别 'cls_num' 相关的四个回归参数：
                #   tx, ty, tw, th 分别代表边界框中心的x偏移量、y偏移量，以及边界框宽度和高度的对数缩放因子。
                # 这四个参数用于调整原始RPN提出的边界框，使其更准确地框定目标对象。
                (tx, ty, tw, th) = P_regr[0, i, 4 * cls_num:4 * (cls_num + 1)]
                # 根据配置的回归标准差对回归信息进行归一化
                # 水平方向的平移归一化,将 'tx' 除以配置中指定的x偏移量的标准差
                tx /= self.config.classifier_regr_std[0]
                # 垂直方向的平移归一化,将 'ty' 除以配置中指定的x偏移量的标准差
                ty /= self.config.classifier_regr_std[1]
                # 宽度缩放归一化,将 'tw' 除以配置中指定的x偏移量的标准差
                tw /= self.config.classifier_regr_std[2]
                # 高度缩放归一化,将 'th' 除以配置中指定的x偏移量的标准差
                th /= self.config.classifier_regr_std[3]

                # ---------------------------------------------------#
                # Step 7: 调整边框位置
                # ---------------------------------------------------#
                # 计算调整后的边框中心点和大小
                # 计算原始边框中心点的横坐标
                cx = x + w / 2.
                # 计算原始边框中心点的纵坐标
                cy = y + h / 2.
                # 根据归一化的平移信息调整中心点的横坐标
                cx1 = tx * w + cx
                # 根据归一化的平移信息调整中心点的纵坐标
                cy1 = ty * h + cy
                # 根据归一化的缩放信息调整边框的宽度
                w1 = math.exp(tw) * w
                # 根据归一化的缩放信息调整边框的高度
                h1 = math.exp(th) * h
                # 计算调整后的边框左上角和右下角坐标
                # 计算调整后的边框左上角的横坐标
                x1 = cx1 - w1 / 2.
                # 计算调整后的边框左上角的纵坐标
                y1 = cy1 - h1 / 2.
                # 计算调整后的边框右下角的横坐标
                x2 = cx1 + w1 / 2
                # 计算调整后的边框右下角的纵坐标
                y2 = cy1 + h1 / 2
                # 将调整后的边框信息加入列表
                # 将边框坐标取整
                # 将左上角横坐标取整
                x1 = int(round(x1))
                # 将左上角纵坐标取整
                y1 = int(round(y1))
                # 将右下角横坐标取整
                x2 = int(round(x2))
                # 将右下角纵坐标取整
                y2 = int(round(y2))
                # 将调整后的边框信息、最大分类概率和标签加入相应的列表
                # 将调整后的边框坐标加入bboxes列表
                bboxes.append([x1, y1, x2, y2])
                # 将最大分类概率加入probs列表
                probs.append(np.max(P_cls[0, i, :]))
                # 将标签加入labels列表
                labels.append(label)
        # 如果没有检测到任何边框，直接返回原始图像
        if len(bboxes) == 0:
            return old_image

        # ---------------------------------------------------#
        # Step 8: 筛选和非极大值抑制（NMS）
        # ---------------------------------------------------#
        # 筛选出其中得分高于confidence的框
        # 将列表转换为NumPy数组，并对边框坐标进行归一化
        # 将标签列表转换为NumPy数组
        labels = np.array(labels)
        # 将概率列表转换为NumPy数组
        probs = np.array(probs)
        # 将边框坐标列表转换为NumPy数组，指定数据类型为float32
        boxes = np.array(bboxes, dtype=np.float32)
        # 对x坐标进行归一化
        # 'boxes[:, 0]' 表示所有边界框的左上角 x 坐标。
        # 通过乘以 'self.config.rpn_stride / width' 来将这些坐标归一化。
        # 这个归一化过程是为了将坐标调整到与模型训练时使用的尺度一致。
        boxes[:, 0] = boxes[:, 0] * self.config.rpn_stride / width
        # 对y坐标进行归一化
        # 类似地，'boxes[:, 1]' 表示所有边界框的左上角 y 坐标。
        # 通过乘以 'self.config.rpn_stride / height' 来将这些坐标归一化。
        boxes[:, 1] = boxes[:, 1] * self.config.rpn_stride / height
        # 对宽度进行归一化
        # 'boxes[:, 2]' 表示所有边界框的宽度。
        # 通过乘以 'self.config.rpn_stride / width' 来将宽度归一化。
        boxes[:, 2] = boxes[:, 2] * self.config.rpn_stride / width
        # 对高度进行归一化
        # 'boxes[:, 3]' 表示所有边界框的高度。
        # 通过乘以 'self.config.rpn_stride / height' 来将高度归一化。
        boxes[:, 3] = boxes[:, 3] * self.config.rpn_stride / height
        # 使用非极大值抑制(NMS)筛选最终的检测结果
        # 使用非极大值抑制 (NMS) 筛选边界框。
        # 'self.bbox_util.nms_for_out' 是执行NMS操作的函数。
        # 'np.array(labels)' 是所有边界框的标签数组。
        # 'np.array(probs)' 是所有边界框的置信度数组。
        # 'np.array(boxes)' 是所有边界框的坐标数组。
        # 'self.num_classes - 1' 指定类别的数量，减1通常是因为最后一个类别是用于表示背景或无对象的。
        # '0.4' 是NMS操作中用于判断边界框重叠的阈值，通常称为IoU（Intersection over Union）阈值。
        # 'results' 会包含NMS后剩余的边界框信息。
        results = np.array(self.bbox_util.nms_for_out(np.array(labels),
                                                      np.array(probs),
                                                      np.array(boxes),
                                                      self.num_classes - 1,
                                                      0.4))
        # 从NMS结果中提取标签、置信度和边框信息
        # 提取标签索引top_label_indices
        top_label_indices = results[:, 0]
        # 提取置信度top_conf
        top_conf = results[:, 1]
        # 提取边框信息boxes
        boxes = results[:, 2:]
        # 对边框坐标进行反归一化，将其转换为原始图像中的坐标
        # 反归一化 x 坐标, 将归一化后的 x 坐标乘以原始图像的宽度
        boxes[:, 0] = boxes[:, 0] * old_width
        # 反归一化 y 坐标,将归一化后的 y 坐标乘以原始图像的高度
        boxes[:, 1] = boxes[:, 1] * old_height
        # 反归一化 宽度,将归一化后的宽度乘以原始图像的宽度
        boxes[:, 2] = boxes[:, 2] * old_width
        # 反归一化 高度,将归一化后的高度乘以原始图像的高度
        boxes[:, 3] = boxes[:, 3] * old_height

        # ---------------------------------------------------#
        # Step 9: 绘制边框和标签
        # ---------------------------------------------------#
        # 加载字体文件，用于绘制文字标签
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        # 计算边框绘制的线条粗细
        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // width
        # 将原始图像赋值给变量image
        image = old_image
        # 遍历每个检测结果，绘制边框和标签
        for i, c in enumerate(top_label_indices):
            # 获取预测类别的名称
            predicted_class = self.class_names[int(c)]
            # 获取预测类别的置信度
            score = top_conf[i]
            # 获取调整后的边框坐标
            left, top, right, bottom = boxes[i]
            # 上边界上移5个像素
            top = top - 5
            # 左边界左移5个像素
            left = left - 5
            # 下边界下移5个像素
            bottom = bottom + 5
            # 右边界右移5个像素
            right = right + 5
            # 对边界坐标进行修正，确保在图像范围内
            # 上边界修正，确保不小于0
            top = max(0, np.floor(top + 0.5).astype('int32'))
            # 左边界修正，确保不小于0
            left = max(0, np.floor(left + 0.5).astype('int32'))
            # 下边界修正，确保不超过图像高度
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            # 右边界修正，确保不超过图像宽度
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
            # 绘制边框和标签
            # 构建标签字符串
            label = '{} {:.2f}'.format(predicted_class, score)
            # 创建绘图对象
            draw = ImageDraw.Draw(image)
            # 获取标签的大小
            label_size = draw.textsize(label, font)
            # 将标签字符串编码为UTF-8格式
            label = label.encode('utf-8')
            # 打印标签(用于调试)
            print(label)
            # 确定文字起始位置
            if top - label_size[1] >= 0:
                # 如果上边界空间足够，文字在上方
                text_origin = np.array([left, top - label_size[1]])
            else:
                # 否则文字在下方，top + 1 是为了避免文字和边框过于紧凑
                text_origin = np.array([left, top + 1])
                # 绘制边框矩形
            # 遍历绘制矩形和标签
            for i in range(thickness):
                # 绘制矩形边框
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[int(c)])  # 绘制矩形边框
            # 填充矩形区域作为背景
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[int(c)])  # 填充背景颜色
            # 绘制标签文字
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)  # 绘制文字标签
            # 释放绘图对象
            del draw
        # ---------------------------------------------------#
        # Step 10: 返回最终图像
        # ---------------------------------------------------#
        # 返回绘制完标签和边框的图像
        return image

    # 关闭 TensorFlow 会话
    def close_session(self):
        self.sess.close()
