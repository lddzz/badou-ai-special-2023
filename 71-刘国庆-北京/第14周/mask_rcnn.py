# 这个Python文件是用于实现Mask R-CNN模型的。
# Mask R-CNN是一种用于目标检测和实例分割的深度学习模型。
# 这个文件中定义了一个名为MASK_RCNN的类,该类包含了Mask R-CNN模型的各种配置和操作。
# 以下是这个文件的主要功能：
# 1. 定义了MASK_RCNN类,该类包含了Mask R-CNN模型的初始化、配置、模型生成和图像检测等方法。
# 2. 在MASK_RCNN类中,定义了一些默认的配置参数,如模型路径、类别文件路径、置信度等。
# 3. get_defaults类方法用于获取默认参数中的某个属性的值。
# 4. get_class方法用于获取所有的分类名称。
# 5. get_config方法用于获取配置对象。
# 6. generate方法用于生成模型。
# 7. detect_image方法用于对输入图像进行检测。
# 8. close_session方法用于关闭Keras会话。
# 总的来说,这个文件的主要作用是实现Mask R-CNN模型的创建、配置和使用。

# 导入用于处理操作系统相关功能的模块 os
import os
# 导入 Keras 模块的后端,用于处理底层操作,并将其命名为 K
import keras.backend as K
# 导入 NumPy 库,并将其命名为 np,用于进行数值计算
import numpy as np
# 从自定义模块 nets.mrcnn 中导入 get_predict_model 函数
from nets.mrcnn import get_predict_model
# 从自定义模块 utils 中导入名为 visualize 的模块
from utils import visualize
# 从自定义模块 utils.anchors 中导入 get_anchors 函数
from utils.anchors import get_anchors
# 从自定义模块 utils.config 中导入 Config 类
from utils.config import Config
# 从自定义模块 utils.utils 中导入 mold_inputs 和 unmold_detections 函数
from utils.utils import mold_inputs, unmold_detections


# 定义名为 MASK_RCNN 的类,继承自 object 类
class MASK_RCNN(object):
    # 定义类变量 defaults,包含模型路径model_path、类别文件路径classes_path、置信度confidence等默认参数
    defaults = {
        "model_path": 'model_data/mask_rcnn_coco.h5',
        "classes_path": 'model_data/coco_classes.txt',
        "confidence": 0.7,
        # 使用coco数据集检测的时候,
        # IMAGE_MIN_DIM=1024,
        # IMAGE_MAX_DIM=1024,
        # RPN_ANCHOR_SCALES=(32, 64, 128, 256, 512)
        "IMAGE_MIN_DIM": 1024,
        "IMAGE_MAX_DIM": 1024,
        "RPN_ANCHOR_SCALES": (32, 64, 128, 256, 512)
        # 在使用自己的数据集进行训练的时候,如果显存不足要调小图片大小
        # 同时要调小anchors
        # "IMAGE_MIN_DIM": 512,
        # "IMAGE_MAX_DIM": 512,
        # "RPN_ANCHOR_SCALES": (16, 32, 64, 128, 256)
    }

    # 定义类方法 get_defaults,用于获取默认参数中的某个属性的值
    @classmethod
    # 定义一个类方法get_defaults,接收两个参数：cls(类本身)和 n(要获取的属性名)
    def get_defaults(cls, n):
        # 检查属性名 n 是否在类变量 defaults 中
        if n in cls._defaults:
            # 如果属性名 n 在 defaults 中,返回 defaults 字典中对应 n 的值
            return cls._defaults[n]
        # 如果属性名 n 不在 defaults 中
        else:
            # 返回一个字符串,提示用户输入的 n 是一个无法识别的属性名
            return n + "是一个无法识别的属性名"

    # ---------------------------------------------------#
    #   初始化Mask-Rcnn
    # ---------------------------------------------------#
    # 类的初始化方法
    def __init__(self):
        # 将对象的属性更新为默认参数的值
        self.__dict__.update(self.defaults)
        # 调用get_class方法获取所有的分类名称,并将其赋值给 class_names 属性
        self.class_names = self.get_class()
        # 获取 Keras 会话,并将其赋值给 sess 属性
        self.sess = K.get_session()
        # 调用get_config方法获取配置对象,并将其赋值给 config 属性
        self.config = self.get_config()
        # 调用 generate 方法,生成模型
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    # 定义一个方法 get_class 用于获取所有的分类名称
    def get_class(self):
        # 将类别文件路径展开用户目录,并赋值给 classes_path
        classes_path = os.path.expanduser(self.classes_path)
        # 打开类别文件,并将其赋值给 f
        with open(classes_path, 'r') as f:
            # 读取所有行,并将其赋值给class_names
            class_names = f.readlines()
        # 对每个类别名称去除首尾空格
        class_names = [c.strip() for c in class_names]
        # 在类别名称列表的首位插入一个名为 "BG" 的类别
        class_names.insert(0, "BG")
        # 返回处理后的类别名称列表
        return class_names

    # ---------------------------------------------------#
    #   获取配置对象
    # ---------------------------------------------------#
    # 定义一个方法 get_config 用于获取配置对象
    def get_config(self):
        # 定义一个内部类 InferenceConfig,继承自 Config 类
        class IngerenceConfig(Config):
            # 在 InferenceConfig 内部类中配置各种参数
            # 配置该子类的一些参数,包括类别数、GPU 数量、每个 GPU 的图像数、检测置信度等
            # 类别数NUM_CLASSES为当前实例的 class_names 属性的长度
            NUM_CLASSES = len(self.class_names)
            # GPU数量GPU_COUNT为 1
            GPU_CONUT = 1
            # 每个 GPU 的图像数IMAGES_PER_GPU为 1
            IMAGES_PER_GPU = 1
            # 检测置信度DETECTION_MIN_CONFIDENCE为当前实例的 confidence 属性值
            DETECTION_MIN_CONFIDENCE = self.confidence
            # 模型的名称NAME为 "shapes"
            NAME = "shapes"
            # RPN 锚点尺度RPN_ANCHOR_SCALES为当前实例的 RPN_ANCHOR_SCALES 属性值
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES
            # 图像最小尺寸IMAGE_MIN_DIM为当前实例的 IMAGE_MIN_DIM 属性值
            IMAGE_MIN_DIM = self.IMAGE_MIN_DIM
            # 图像最大尺寸IMAGE_MAX_DIM为当前实例的 IMAGE_MAX_DIM 属性值
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM

        # 创建 InferenceConfig 的实例
        config = IngerenceConfig()
        # 显示配置信息
        config.display()
        # 返回配置对象
        return config

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    # 定义一个方法 generate 用于生成模型
    def generate(self):
        # 将模型路径展开为绝对路径
        model_path = os.path.expanduser(self.model_path)
        # 断言检查模型路径是否以 ".h5" 结尾,如果不是,则抛出异常
        assert model_path.endswith(".h5"), "Keras 模型或者权重文件必须是 .h5 文件"
        # 计算类别名称列表的长度,即类别的数量,并赋值给 num_classes 属性
        self.num_classes = len(self.class_names)
        # 调用 get_predict_model 函数,传入配置对象,生成模型,并赋值给 model 属性
        self.model = get_predict_model(self.config)
        # 调用模型的 load_weights 方法,加载权重文件,by_name 参数设置为 True,表示按层名称进行权重加载
        self.model.load_weights(model_path, by_name=True)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    # 定义一个方法 detect_image 用于对输入图像进行检测
    def detect_image(self, image):
        # 将输入图像转换为 NumPy 数组,并将其放入一个列表中
        image = [np.array(image)]
        # 调用 mold_inputs 函数,
        # 传入配置对象和图像列表,
        # 返回处理后的图像molded_images、图像元数据image_metas和窗口windows
        molded_images, image_metas, windows = mold_inputs(self.config, image)
        # 获取经过预处理处理后的图像的形状(高度、宽度和颜色通道数)
        image_shape = molded_images[0].shape
        # 调用 get_anchors 函数,传入配置对象和图像形状,获取锚点anchors
        anchors = get_anchors(self.config, image_shape)
        # 将锚点数组广播到一个新的形状,新的形状的第一个维度是 1,后面的维度是锚点数组原来的形状
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)
        # 调用模型的 predict 方法,
        # 传入预处理后的图像molded_images图像元数据image_metas和锚点anchors,
        # verbose=0 表示在预测过程中不输出详细的日志信息
        # 获取detections检测结果,_, _, mrcnn_mask预测掩膜,_, _, _ 等,其他的 _ 是用来接收预测结果中不需要的部分
        # 使用模型对输入的图像进行预测,并获取预测的检测结果和掩膜
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.model.predict([molded_images, image_metas, anchors], verbose=0)
        # 调用 unmold_detections 函数
        # 传入检测结果detections[0]
        # 预测的掩膜mrcnn_mask[0]
        # 原始图像的形状image[0].shape
        # 处理后的图像的形状molded_images[0].shape
        # 处理后的图像的窗口 windows[0]
        # 获取最终的检测结果,
        # 包括每个检测到的对象的位置(final_rois)、类别(final_class_ids)、得分(final_scores)和形状(final_masks)
        final_rois, final_class_ids, final_scores, final_masks = \
            unmold_detections(detections[0], mrcnn_mask[0],
                              image[0].shape, molded_images[0].shape,
                              windows[0])
        # 将最终的区域、类别 ID、分数和掩膜放入一个字典中
        # 创建一个字典，包含了 Mask R-CNN 模型预测的结果
        # 检测到的每个对象的位置信息 final_rois放置在 rois 键中
        # 检测到的每个对象的类别 ID final_class_ids放置在 class_ids 键中
        # 检测到的每个对象的得分 final_scores放置在 scores 键中
        # 检测到的每个对象的掩膜 final_masks放置在 masks 键中
        r = {
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }
        # 调用 visualize 模块的 display_instances 函数,显示检测结果
        # 传入
        # 原始图像image[0]
        # 区域r['rois']
        # 掩膜r['masks']
        # 类别 IDr['class_ids']
        # 类别名称self.class_names
        # 分数r['scores']
        visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'],
                                    self.class_names, r['scores'])

    # 定义一个方法 close_session 用于关闭 Keras 会话
    def close_session(self):
        # 调用 Keras 会话的 close 方法
        self.sess.close()
