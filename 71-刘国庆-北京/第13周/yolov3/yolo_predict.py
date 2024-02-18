# 导入用于与操作系统交互的 os 模块
import os
# 导入配置参数的模块
import config
# 导入用于生成随机数的 random 模块
import random
# 导入颜色空间转换相关的 colorsys 模块
import colorsys
# 导入 NumPy 库并用别名 np 引用,用于进行数组和矩阵操作
import numpy as np
# 导入 TensorFlow 深度学习框架,并用别名 tf 引用
import tensorflow as tf
# 从 model.yolo3_model 模块中导入名为 yolo 的 YOLO 模型
from model.yolo3_model import yolo


# 定义名为 yolo_predictor 的类
class yolo_predictor:
    # 初始化方法
    # 接收目标检测为物体的阈值obj_threshold
    # 非极大值抑制阈值nms_threshold
    # 类别文件路径classes_path
    # 先验框文件路径作为参数anchors_path
    def __init__(self, obj_threshold, nms_threshold, classes_path, anchors_path):
        # 设置目标检测为物体的阈值
        self.obj_threshold = obj_threshold
        # 设置非极大值抑制阈值
        self.nms_threshold = nms_threshold
        # 预读取类别文件路径
        self.classes_path = classes_path
        # 预读取先验框文件路径
        self.anchors_path = anchors_path
        # 调用get_class方法读取种类名称
        self.class_names = self.get_class()
        # 调用get_anchors方法读取先验框
        self.anchors = self.get_anchors()
        # 画框框用,生成一组用于标记不同类别的颜色
        # 创建一个 HSV 颜色元组的列表 hsv_tuples。
        # 列表中的每个元组代表一个颜色,用 HSV 色彩空间表示。
        # x / len(self.class_names) 计算每个类别对应的色调(Hue)值,其范围在 0 到 1 之间。
        # 通过 for x in range(len(self.class_names)) 遍历所有类别。
        # 每个元组中的 1. 表示饱和度(Saturation)和明度(Value)都设置为最大值 1。
        # 这样做的目的是为每个类别生成一个明亮鲜艳的颜色。
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        # 转换为RGB颜色值
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # 将RGB颜色值的元组列表进行缩放,确保每个通道值在0-255范围内
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # 设定随机数种子,用于打乱颜色顺序
        random.seed(10101)
        random.shuffle(self.colors)
        # 恢复随机数种子状态
        random.seed(None)

    # 读取类别名称函数get_class
    def get_class(self):
        # 获取完整的类别文件路径classes_path
        class_path = os.path.expanduser(self.classes_path)
        # 使用 with 语句打开文件,确保在使用完文件后正确关闭
        with open(class_path) as f:
            # 读取文件的所有行,将其存储在列表 class_names 中
            class_names = f.readlines()
        # 使用列表推导式去除每个类别名称前后可能存在的空白字符
        class_names = [c.strip() for c in class_names]
        # 将处理后的类别名称列表作为方法的返回值
        return class_names

    # 读取anchors数据函数get_anchors
    def get_anchors(self):
        # 获取完整的anchors文件路径
        anchors_path = os.path.expanduser(self.anchors_path)
        # 使用 with 语句打开文件,确保在使用完文件后正确关闭
        with open(anchors_path) as f:
            # 读取文件的第一行,包含anchors数据,放在anchors
            anchors = f.readline()
            # 将anchors数据拆分为浮点数列表
            anchors = [float(x) for x in anchors.split(',')]
            # 将列表转换为NumPy数组,并重新形状为二维数组
            anchors = np.array(anchors).reshape(-1, 2)
        # 将处理后的anchors数组作为方法的返回值
        return anchors

    # ---------------------------------------#
    #   对三个特征层解码
    #   进行排序并进行非极大抑制
    # ---------------------------------------#

    # boxes_and_scores函数将预测出的box坐标转换为对应原图的坐标,然后计算每个box的分数
    # 输入:
    # feats: yolo输出的featuremap
    # anchors: anchor的位置
    # class_num: 类别数目
    # input_shape: 输入大小
    # image_shape: 图片大小
    # 返回:
    # boxes: 物体框的位置
    # boxes_scores: 物体框的分数,为置信度和类别概率的乘积
    def boxes_and_scores(self, feats, anchors, class_num, input_shape, image_shape):
        # 调用get_feats方法
        # 输入feats: yolo输出的featuremap,anchors: anchor的位置,class_num: 类别数目,input_shape: 输入大小
        # 获取框的中心坐标box_xy,宽高box_wh,置信度box_confidence和类别概率box_class_probs
        box_xy, box_wh, box_confidence, box_class_probs = self.get_feats(feats, anchors, class_num, input_shape)
        # 调用correct_boxes方法校正框在原图上的位置
        # 输入框的中心坐标box_xy,宽高box_wh,输入大小input_shape,图片大小image_shape
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        # 调用reshape方法,调整框的形状为[-1, 4],确保每个框都有四个坐标值
        boxes = tf.reshape(boxes, [-1, 4])
        # 计算框的得分,即置信度乘以类别概率:box_confidence * box_class_probs
        scores = box_confidence * box_class_probs
        # 调用reshape方法,调整框的得分形状为[-1, classes_num],确保每个框都有对应的类别得分
        scores = tf.reshape(scores, [-1, class_num])
        # 返回处理后的物体框的位置和得分
        return boxes, scores

    # 获得在原图上框的位置
    # 计算物体框预测坐标在原图中的位置坐标
    # 输入:
    # box_xy: 物体框左上角坐标,box_wh: 物体框的宽高,input_shape: 输入的大小,image_shape: 图片的大小
    # 返回:
    # boxes: 物体框的位置
    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        # 将box_xy和box_wh中的坐标顺序反转
        # 将box_xy的坐标顺序反转,由 (x, y) 转为 (y, x)
        box_yx = box_xy[..., ::-1]
        # 将box_wh的坐标顺序反转,由 (width, height) 转为 (height, width)
        box_hw = box_wh[..., ::-1]
        # 将输入大小input_shape转换为浮点数float32
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        # 将图片大小image_shape转换为浮点数float32
        image_shape = tf.cast(image_shape, dtype=tf.float32)
        # 计算新的形状,通过对原始图片大小进行调整,保持宽高比例不变
        # 参数说明：
        #   image_shape: 原始图片的大小,形状为 [height, width]
        #   input_shape: 模型输入的大小,形状为 [target_height, target_width]
        #   tf.reduce_min(input_shape / image_shape): 计算宽高比例的最小值,用于调整原始图片大小
        #   image_shape * tf.reduce_min(input_shape / image_shape): 根据比例调整原始图片的宽和高
        #   tf.round(): 对计算结果进行四舍五入,得到整数形状
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

        # 计算偏移量和缩放比例
        # 计算偏移量offset,使调整后的图片位于输入图片中心
        # 参数说明：
        #   input_shape: 模型输入的大小,形状为 [target_height, target_width]
        #   new_shape: 根据比例调整后的原始图片大小,形状为 [new_height, new_width]
        #   (input_shape - new_shape) / 2.: 计算偏移量offset,确保调整后的图片在输入图片中心
        #   input_shape / new_shape: 用于规范化偏移量,确保在 [0, 1] 范围内
        offset = (input_shape - new_shape) / 2. / input_shape
        # 计算缩放比例scale,确保物体框坐标的正确映射
        # 参数说明：
        #   input_shape: 模型输入的大小,形状为 [target_height, target_width]
        #   new_shape: 根据比例调整后的原始图片大小,形状为 [new_height, new_width]
        #   input_shape / new_shape: 计算缩放比例scale,确保物体框坐标的正确映射
        scale = input_shape / new_shape
        # 根据偏移量和缩放比例调整box_xy和box_hw
        # 根据偏移量和缩放比例调整物体框的左上角坐标
        # 参数说明：
        #   box_yx: 物体框左上角坐标,形状为 [batch_size, grid_size, grid_size, num_boxes, 2]
        #   offset: 偏移量,确保调整后的图片位于输入图片中心,形状为 [1, 1, 1, 1, 2]
        #   scale: 缩放比例,确保物体框坐标的正确映射,形状为 [1, 1, 1, 1, 2]
        box_yx = (box_yx - offset) * scale
        # 根据缩放比例调整物体框的宽高
        # 参数说明：
        #   box_hw: 物体框的宽高,形状为 [batch_size, grid_size, grid_size, num_boxes, 2]
        #   scale: 缩放比例,确保物体框坐标的正确映射,形状为 [1, 1, 1, 1, 2]
        box_hw *= scale
        # 计算box的最小和最大坐标
        # 计算物体框的最小坐标和最大坐标
        # 参数说明：
        #   box_yx: 调整后的物体框左上角坐标,形状为 [batch_size, grid_size, grid_size, num_boxes, 2]
        #   box_hw: 调整后的物体框宽高,形状为 [batch_size, grid_size, grid_size, num_boxes, 2]
        #   box_mins: 计算得到的物体框的最小坐标,形状同 box_yx
        #   box_maxes: 计算得到的物体框的最大坐标,形状同 box_yx
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        # 将坐标拼接为boxes
        # 将物体框的最小坐标和最大坐标拼接为boxes
        # 参数说明：
        #   box_mins: 物体框的最小坐标,形状为 [batch_size, grid_size, grid_size, num_boxes, 2]
        #   box_maxes: 物体框的最大坐标,形状为 [batch_size, grid_size, grid_size, num_boxes, 2]
        #   boxes: 拼接得到的物体框的坐标,形状为 [batch_size, grid_size, grid_size, num_boxes, 4]
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        # 将boxes坐标映射回原始图片大小
        # 参数说明：
        #   boxes: 拼接得到的物体框的坐标,形状为 [batch_size, grid_size, grid_size, num_boxes, 4]
        #   image_shape: 原始图片的大小,形状为 [2],包含高度和宽度信息
        #   tf.concat([image_shape, image_shape], axis=-1): 将原始图片大小复制一份,形状为 [4],方便与boxes进行逐元素相乘
        #   boxes *= tf.concat([image_shape, image_shape], axis=-1): 将boxes坐标映射回原始图片大小
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        # 返回物体框的位置
        return boxes

    # 其实是解码的过程get_feats
    #  根据yolo最后一层的输出确定bounding box
    # 输入:
    # feats: yolo模型最后一层输出,anchors: anchors的位置,num_classes: 类别数量,input_shape: 输入大小
    # 返回:
    # 框的中心坐标box_xy,宽高box_wh,置信度box_confidence,类别概率box_class_probs
    def get_feats(self, feats, anchors, num_classes, input_shape):
        # 获取anchors的数量num_anchors
        num_anchors = len(anchors)
        # 调用constant方法将anchors转换为张量,数据类型为32位浮点数,并重新形状为 [1, 1, 1, num_anchors, 2]
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        # 获取feats的宽度和高度信息grid_size
        grid_size = tf.shape(feats)[1:3]
        # 将feats重新形状为 [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5]作为predictions
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 构建13*13*1*2的矩阵,对应每个格子加上对应的坐标
        # 创建包含垂直方向网格坐标的矩阵 grid_y
        # 通过 tf.range 创建一个包含 0 到 grid_size[0] - 1 的一维张量
        # 复制第一个维度(batch_size)的数据,使其在第二维度(网格的垂直维度)上重复 grid_size[1] 次,其他维度不变
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])

        # 创建包含水平方向网格坐标的矩阵 grid_x
        # 通过 tf.range 创建一个包含 0 到 grid_size[1] - 1 的一维张量
        # 复制第一个维度(网格的水平维度)的数据,使其在第二维度(batch_size)上重复 grid_size[0] 次,其他维度不变
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        # 将x和y坐标矩阵合并为grid
        # 将 x 和 y 坐标的网格合并为一个完整的网格矩阵
        # 张量列表,包含水平和垂直方向的网格坐标矩阵
        # 参数说明：axis=-1 表示沿着最后一个轴(列方向)进行连接
        grid = tf.concat([grid_x, grid_y], axis=-1)
        # 将grid的数据类型转换为float32
        grid = tf.cast(grid, tf.float32)
        # 计算预测框的中心坐标,并进行归一化
        # 使用 sigmoid 函数将 predictions 中的前两个通道(x, y坐标)进行激活,计算预测框的中心坐标,
        # 将网格尺寸反转,以便在后续计算中正确地进行归一化,将数据类型转换为 float32
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 计算预测框的宽度和高度,并进行归一化
        # # 使用指数函数将模型预测的第三和第四个通道(w, h)进行解码,得到原始的框宽度和高度信息
        # 将输入图像的形状反转,以便在后续计算中正确地进行归一化,将数据类型转换为 float32
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        # 计算预测框的置信度(是否包含目标)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        # 计算预测框的类别概率
        box_class_prob = tf.sigmoid(predictions[..., 5:])
        # 返回归一化后的框的中心坐标,宽高,置信度和类别概率
        return box_xy, box_wh, box_confidence, box_class_prob

    # 根据Yolo模型的输出进行非极大值抑制,获取最后的物体检测框和物体检测类别
    # 输入:
    # yolo_outputs: yolo模型输出,image_shape: 图片的大小,max_boxes: 最大box数量为20
    # 返回:
    # boxes_: 物体框的位置,scores_: 物体类别的概率,classes_: 物体类别
    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        # 定义每个特征层对应的先验框的索引anchor_mask
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # 初始化空列表,用于存储预测框的坐标boxes和相应的分数box_scores
        boxes = []
        box_scores = []
        # 计算输入形状input_shape,这里使用了 TensorFlow 的 tf.shape 函数
        # 获取 yolo_outputs 的第一个元素的形状input_shape,然后取其第一和第二个元素,并乘以32
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
        # 对三个特征层的输出获取每个预测box坐标和box的分数,score = 置信度x类别概率
        # 对每个特征层的输出进行循环处理
        for i in range(len(yolo_outputs)):
            # 调用boxes_and_scores方法,# 获取每个特征层的预测框坐标_boxes和分数_box_scores,传递以下参数
            # 特征层输出yolo_outputs,
            # 对应的锚点self.anchors[anchor_mask[i]],
            # 类别数len(self.class_names),
            # 输入形状input_shape,
            # 图像形状image_shape
            _boxes, _box_scores = self.boxes_and_scores(
                yolo_outputs[i],
                self.anchors[anchor_mask[i]],
                len(self.class_names),
                input_shape,
                image_shape
            )
            # 将得到的预测框坐标和分数分别加入列表boxes和box_scores
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        # 将三个特征层的结果合并
        # 在垂直方向上拼接列表中的元素,得到所有特征层的预测框的坐标
        boxes = tf.concat(boxes, axis=0)
        # 将三个特征层的预测框分数在垂直方向上合并为一个列表
        box_scores = tf.concat(box_scores, axis=0)
        # 过滤掉得分低于obj_threshold的框
        mask = box_scores >= self.obj_threshold
        # 根据max_boxes调用constan方法创建整型常量张量max_boxes_tensor,表示最大的框数
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        # 初始化空列表,用于存储过滤后的框的坐标boxes_,分数scores_和类别classes_
        boxes_ = []
        scores_ = []
        classes_ = []
        # 对每一个类进行判断
        for c in range(len(self.class_names)):
            # 取出所有类为c的box,调用boolean_mask通过布尔掩码将得分低于obj_threshold的框过滤掉
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为c的分数,通过布尔掩码将得分低于obj_threshold的框过滤掉
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # 非极大抑制,获取经过非极大抑制后保留的框的索引
            # 进行非极大抑制,获取经过非极大抑制tf.image.non_max_suppression后保留的框的索引
            # 预测框坐标class_boxes,预测框对应的分数class_box_scores,最大保留的框数max_boxes_tensor,IOU 阈值,用于判断两个框是否重叠
            nms_index = tf.image.non_max_suppression(
                class_boxes,
                class_box_scores,
                max_boxes_tensor,
                iou_threshold=self.nms_threshold
            )
            # 获取非极大抑制的结果,通过非极大抑制后的索引获取对应的框
            # 调用gather方法,从 class_boxes 中收集与经过非极大抑制后保留的框的索引 nms_index 相对应的框坐标
            class_boxes = tf.gather(class_boxes, nms_index)
            # 根据非极大抑制的结果,获取对应的框的分数
            # 调用gather方法,从 class_box_scores 中收集与经过非极大抑制后保留的框的索引 nms_index 相对应的框分数
            class_box_scores = tf.gather(class_box_scores, nms_index)
            # 为每个类别创建一个与分数相同长度的类别张量
            classes = tf.ones_like(class_box_scores, 'int32') * c
            # 将每个类别的非极大抑制结果加入对应的列表
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        # 将每个类别的非极大抑制结果在垂直方向上合并为一个列表
        # boxes_ 存储了所有类别的保留框的坐标,scores_ 存储了对应的分数,classes_ 存储了类别标签
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        # 返回合并后的框的坐标,分数,类别类别标签
        return boxes_, scores_, classes_

    # ---------------------------------------#
    #   predict用于预测,分三步
    #   1,建立yolo对象
    #   2,获得预测结果
    #   3,对预测结果进行处理
    # ---------------------------------------#
    # 构建预测模型
    # 输入:
    # inputs: 处理之后的输入图片,image_shape: 图像原始大小
    # 返回:
    # boxes: 物体框坐标,scores: 物体概率值,classes: 物体类别
    # 定义一个名为 predict 的方法,接受 self(类实例),inputs(输入数据),image_shape(图像形状)三个参数
    def predict(self, inputs, image_shape):
        # 创建一个新的 YOLO 模型实例,传递一些参数：
        # config.norm_epsilon,config.norm_decay,self.anchors_path,self.classes_path和 pre_train=False
        model = yolo(
            config.norm_epsilon,
            config.norm_decay,
            self.anchors_path,
            self.classes_path,
            pre_train=False
        )

        # 使用 YOLO 模型的 yolo_inference 方法获取网络的预测结果
        # 传递参数：
        # inputs(输入数据),
        # num_anchors(锚框数量除以3,即):config.num_anchors // 3
        # num_classes(类别数量):config.num_classes,
        # training=False(推理模式)
        output = model.yolo_inference(
            inputs,
            config.num_anchors // 3,
            config.num_classes,
            training=False
        )

        # 使用 eval 方法进行非极大值抑制,获取最终的物体框坐标,概率值和类别
        # 传递参数：output(预测结果),image_shape(图像形状),max_boxes=20(最大框数)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)

        # 返回物体框坐标,概率值和类别
        return boxes, scores, classes
