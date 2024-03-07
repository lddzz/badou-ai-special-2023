# 导入colorsys库，它提供了将颜色从一个颜色系统转换到另一个颜色系统的函数
import colorsys
# 导入onnx库
import onnx
# 导入os库，它提供了许多与操作系统交互的函数
import os
# 导入time库，它提供了时间相关的函数
import time
# 导入cv2库，这是一个开源的计算机视觉库，提供了大量的图像和视频处理功能
import cv2
# 导入matplotlib.pyplot库，并将其别名为plt，这是一个用于创建图表和可视化数据的Python库
import matplotlib.pyplot as plt
# 导入numpy库，并将其别名为np，它是一个用于处理数组的Python库
import numpy as np
# 导入torch库，它是一个开源的机器学习库，提供了强大的张量计算（如数组计算）和深度学习算法
import torch
# 从torch库中导入nn模块，它提供了构建神经网络的各种组件
import torch.nn as nn
# 从dcmtracking.detection.yolov5.nets.yolo模块中导入YoloBody类，这是YOLO模型的实现
from dcmtracking.detection.yolov5.nets.yolo import YoloBody
# 从dcmtracking.detection.yolov5.utils.utils模块中导入一些函数，这些函数用于图像预处理和模型配置的显示
from dcmtracking.detection.yolov5.utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                                                      resize_image, show_config)
# 从dcmtracking.detection.yolov5.utils.utils_bbox模块中导入DecodeBox类，这是一个用于解码YOLO模型输出的工具
from dcmtracking.detection.yolov5.utils.utils_bbox import DecodeBox


# 定义YOLO类
class YOLO(object):
    # 定义默认参数
    # 定义一个名为 defaults 的字典，存储YOLO模型的默认参数
    defaults = {
        # 模型路径，指定了YOLO模型的.pth文件的位置
        "model_path": 'dcmtracking/detection/yolov5/model_data/yolov5_s_v6.1.pth',
        # 类别路径，指定了一个.txt文件的位置，该文件包含了模型可以识别的所有类别的名称
        "classes_path": 'dcmtracking/detection/yolov5/model_data/coco_classes.txt',
        # 锚点路径，指定了一个.txt文件的位置，该文件包含了YOLO模型使用的所有锚点的大小
        "anchors_path": 'dcmtracking/detection/yolov5/model_data/yolo_anchors.txt',
        # 锚点掩码，这是一个列表，包含了YOLO模型在每个尺度上使用的锚点的索引
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # 输入形状，这是一个列表，指定了模型输入图像的高度和宽度
        "input_shape": [640, 640],
        # 模型大小，这是一个字符串，指定了YOLO模型的大小，'s'代表小型模型
        "phi": 's',
        # 置信度阈值，这是一个浮点数，用于在非极大值抑制过程中过滤掉置信度较低的预测框
        "confidence": 0.5,
        # 非极大值抑制的IOU阈值，这是一个浮点数，用于在非极大值抑制过程中判断两个预测框是否重叠
        "nms_iou": 0.3,
        # 是否使用letterbox_image，这是一个布尔值，如果为True，则在预处理图像时会保持原始图像的纵横比
        "letterbox_image": True,
        # 是否使用CUDA，这是一个布尔值，如果为True并且系统支持CUDA，则模型会在GPU上运行
        "cuda": torch.cuda.is_available(),
    }

    # 初始化函数
    # 定义初始化方法，接受任意数量的关键字参数
    def __init__(self, **kwargs):
        # 使用默认参数更新实例的字典，这样如果在创建实例时没有提供某些参数，就会使用默认参数
        self.__dict__.update(self.defaults)
        # 遍历传入的关键字参数
        for name, value in kwargs.items():
            # 使用setattr函数设置实例的属性，name是属性名，value是属性值
            setattr(self, name, value)
            # 更新默认参数，这样如果以后再创建实例时没有提供这个参数，就会使用新的默认值
            self.defaults[name] = value
        # 调用get_classes函数获取类别名和类别数，这两个值分别赋给self.class_names和self.num_classes
        self.class_names, self.num_classes = get_classes(self.classes_path)
        # 调用get_anchors函数获取锚点和锚点数，这两个值分别赋给self.anchors和self.num_anchors
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        # 创建一个DecodeBox实例，输入参数为锚点、类别数、输入形状和锚点掩码，将这个实例赋给self.bbox_util
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)
        # 调用get_colors方法获取颜色，将返回的颜色赋给self.colors
        self.colors = self.get_colors()
        # 调用generate方法生成模型
        self.generate()
        # 调用show_config函数显示配置，输入参数为默认参数
        show_config(**self.defaults)

    # 获取默认参数的函数
    # 使用 @classmethod 装饰器，表示这是一个类方法，类方法的第一个参数是类本身，通常命名为 cls
    @classmethod
    # 定义一个名为 get_defaults 的类方法，接受一个名为 n 的参数
    def get_defaults(cls, n):
        # 检查参数 n 是否在类的默认参数字典 defaults 中
        if n in cls.defaults:
            # 如果在，则返回对应的值
            return cls.defaults[n]
        else:
            # 如果不在，则返回一个字符串，表示无法找到名为 n 的参数
            return f"无法找到名字为{n}的参数！"

    # 获取颜色的函数
    # 定义一个名为 get_colors 的方法，接受 self(类实例)一个参数
    def get_colors(self):
        # 创建一个HSV颜色元组列表，列表的长度等于目标类别的数量，每个元组的H（色相）值由类别的索引决定，S（饱和度）和V（明度）值都为1
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        # 将HSV颜色元组列表转换为RGB颜色元组列表，使用colorsys库的hsv_to_rgb函数进行转换
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # 将RGB颜色元组列表中的每个元素乘以255并转换为整数，因为在计算机图像中，颜色的范围通常是0到255
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        # 返回RGB颜色元组列表
        return colors

    # 生成模型的函数
    # 定义一个名为 generate 的方法，接受一个名为 onnx 的参数，默认值为 False
    def generate(self, onnx=False):
        # 创建YOLO模型，输入参数为锚点掩码、类别数和模型大小
        self.net = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        # 判断是否支持CUDA，如果支持则使用CUDA，否则使用CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载预训练模型，模型路径为 self.model_path，加载的设备为 device
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        # 将模型设置为评估模式，这意味着在此模式下，模型的所有dropout和batchnorm层都会工作在评估模式下
        self.net = self.net.eval()
        # 打印模型加载信息
        print('{} model, and classes loaded.'.format(self.model_path))
        # 如果不是导出onnx模型且支持CUDA，则将模型设置为数据并行模式，并将模型移动到GPU
        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # 检测图像的函数
    # 定义一个名为 detect_image 的方法，接受 self(类实例), image(输入图像), crop(是否裁剪目标，默认为False), count(是否计数，默认为False)四个参数
    def detect_image(self, image, crop=False, count=False):
        # 调用 get_detection_results 方法，获取图像的检测结果
        top_label, top_boxes, top_conf = self.get_detection_results(image)
        # 如果需要计数，则调用 count_classes 方法，对目标的类别进行计数
        if count:
            self.count_classes(top_label)
        # 如果需要裁剪，则调用 crop_objects 方法，对目标进行裁剪
        if crop:
            self.crop_objects(image, top_boxes)
        # 返回目标的标签、框的坐标和置信度
        return top_label, top_boxes, top_conf

    # 检测图像的函数（ppe）
    # 定义一个名为 detect_image_ppe 的方法，接受 self(类实例), image(输入图像), crop(是否裁剪目标，默认为False), count(是否计数，默认为False)四个参数
    def detect_image_ppe(self, image, crop=False, count=False):
        # 调用 get_detection_results 方法，获取图像的检测结果
        top_label, top_boxes, top_conf = self.get_detection_results(image)
        # 如果需要计数，则调用 count_classes 方法，对目标的类别进行计数
        if count:
            self.count_classes(top_label)
        # 如果需要裁剪，则调用  方法，对目标进行裁剪
        if crop:
            self.crop_objects(image, top_boxes)
        # 返回处理后的图像
        return image

    # 获取FPS的函数
    # 定义一个名为 get_FPS 的方法，接受 self(类实例), image(输入图像), test_interval(测试间隔)三个参数
    def get_FPS(self, image, test_interval):
        # 获取输入图像的形状（高度和宽度）
        image_shape = np.array(np.shape(image)[0:2])
        # 将输入图像转换为RGB格式
        image = cvtColor(image)
        # 对输入图像进行resize，使其符合模型的输入要求
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 对图像数据进行预处理，并增加一个维度，使其符合模型的输入要求
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # 使用torch.no_grad()来关闭梯度计算，以减少内存使用并加速计算
        with torch.no_grad():
            # 将numpy数组转换为torch张量
            images = torch.from_numpy(image_data)
            # 如果使用CUDA，则将数据移动到GPU
            if self.cuda:
                images = images.cuda()
            # 将图像数据输入到模型中，获取模型的输出
            outputs = self.net(images)
            # 使用解码框工具对模型的输出进行解码
            outputs = self.bbox_util.decode_box(outputs)
            # 对解码后的输出进行非极大值抑制，以去除冗余的检测框
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
        # 获取当前时间
        t1 = time.time()
        # 进行test_interval次检测，以计算平均FPS
        for _ in range(test_interval):
            with torch.no_grad():
                # 将图像数据输入到模型中，获取模型的输出
                outputs = self.net(images)
                # 使用解码框工具对模型的输出进行解码
                outputs = self.bbox_util.decode_box(outputs)
                # 对解码后的输出进行非极大值抑制，以去除冗余的检测框
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)
        # 获取当前时间
        t2 = time.time()
        # 计算FPS，即每秒可以处理的帧数
        tact_time = (t2 - t1) / test_interval
        # 返回FPS
        return tact_time

    # 检测热图的函数
    # 定义一个名为 detect_heatmap 的方法，接受 self(类实例), image(输入图像), heatmap_save_path(热图保存路径)三个参数
    def detect_heatmap(self, image, heatmap_save_path):
        # 定义sigmoid函数
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        # 将输入图像转换为RGB格式
        image = cvtColor(image)
        # 对输入图像进行resize，使其符合模型的输入要求
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 对图像数据进行预处理，并增加一个维度，使其符合模型的输入要求
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # 使用torch.no_grad()来关闭梯度计算，以减少内存使用并加速计算
        with torch.no_grad():
            # 将numpy数组转换为torch张量
            images = torch.from_numpy(image_data)
            # 如果使用CUDA，则将数据移动到GPU
            if self.cuda:
                images = images.cuda()
            # 将图像数据输入到模型中，获取模型的输出
            outputs = self.net(images)
        # 显示原始图像
        plt.imshow(image, alpha=1)
        # 关闭坐标轴
        plt.axis('off')
        # 创建一个全零的掩码，形状与输入图像相同
        mask = np.zeros((image.size[1], image.size[0]))
        # 遍历模型的输出
        for sub_output in outputs:
            # 将torch张量转换为numpy数组
            sub_output = sub_output.cpu().numpy()
            # 获取输出的形状
            b, c, h, w = np.shape(sub_output)
            # 调整输出的形状
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            # 计算得分
            score = np.max(sigmoid(sub_output[..., 4]), -1)
            # 调整得分的大小，使其与输入图像的大小相同
            score = cv2.resize(score, (image.size[0], image.size[1]))
            # 归一化得分
            normed_score = (score * 255).astype('uint8')
            # 更新掩码
            mask = np.maximum(mask, normed_score)
        # 显示掩码
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")
        # 关闭坐标轴
        plt.axis('off')
        # 调整子图布局
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # 设置边距
        plt.margins(0, 0)
        # 保存图像
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches=-0.1)
        # 打印保存信息
        print("Save to the " + heatmap_save_path)
        # 显示图像
        plt.show()

    # 转换为ONNX的函数
    # 定义一个名为 convert_to_onnx 的方法，接受 self(类实例), simplify(是否简化模型), model_path(模型保存路径)三个参数
    def convert_to_onnx(self, simplify, model_path):
        # 调用generate方法生成模型，参数onnx设为True
        self.generate(onnx=True)
        # 创建一个全零的torch张量，形状为[1, 3, *self.input_shape]，并将其移动到CPU
        im = torch.zeros(1, 3, *self.input_shape).to('cpu')
        # 定义模型的输入层名称
        input_layer_names = ["images"]
        # 定义模型的输出层名称
        output_layer_names = ["output"]
        # 打印开始导出的信息，包括onnx的版本
        print(f'Starting export with onnx {onnx.__version__}.')
        # 使用torch.onnx.export方法将模型导出为ONNX格式
        torch.onnx.export(self.net, im, f=model_path, verbose=False, opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True, input_names=input_layer_names, output_names=output_layer_names,
                          dynamic_axes=None)
        # 加载导出的ONNX模型
        model_onnx = onnx.load(model_path)
        # 检查模型是否符合ONNX的规范
        onnx.checker.check_model(model_onnx)
        # 如果simplify为True，则进行模型简化
        if simplify:
            # 导入onnxsim库
            import onnxsim
            # 打印开始简化的信息，包括onnx-simplifier的版本
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            # 使用onnxsim.simplify方法简化模型
            model_onnx, check = onnxsim.simplify(model_onnx, dynamic_input_shape=False, input_shapes=None)
            # 检查简化结果，如果check为False，则抛出异常
            assert check, 'assert check failed'
            # 保存简化后的模型
            onnx.save(model_onnx, model_path)
        # 打印模型保存的路径
        print('Onnx model save as {}'.format(model_path))

    # 获取MAP文本的函数
    # 定义一个名为 get_map_txt 的方法，接受 self(类实例), image_id(图像ID), image(输入图像), class_names(类别名称), map_out_path(输出路径)五个参数
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        # 调用 get_detection_results 方法，获取图像的检测结果
        top_label, top_boxes, top_conf = self.get_detection_results(image)
        # 如果没有检测到任何目标，则直接返回
        if top_label is None:
            return
        # 打开一个新的文本文件，用于存储检测结果
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        # 遍历每一个检测到的目标
        for i, c in list(enumerate(top_label)):
            # 获取目标的类别名称
            predicted_class = self.class_names[int(c)]
            # 如果目标的类别不在给定的类别列表中，则跳过当前循环
            if predicted_class not in class_names:
                continue
            # 获取目标的框的坐标
            box = top_boxes[i]
            # 获取目标的置信度
            score = str(top_conf[i])
            # 获取框的上、左、下、右四个边的坐标
            top, left, bottom, right = box
            # 将目标的类别名称、置信度和框的坐标写入到文本文件中
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        # 关闭文本文件
        f.close()

    # 获取检测结果的函数
    # 定义一个名为 get_detection_results 的方法，接受 self(类实例) 和 image(输入图像)两个参数
    def get_detection_results(self, image):
        # 获取输入图像的形状（高度和宽度）
        image_shape = np.array(np.shape(image)[0:2])
        # 将输入图像转换为RGB格式
        image = cvtColor(image)
        # 对输入图像进行resize，使其符合模型的输入要求
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 对图像数据进行预处理，并增加一个维度，使其符合模型的输入要求
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # 使用torch.no_grad()来关闭梯度计算，以减少内存使用并加速计算
        with torch.no_grad():
            # 将numpy数组转换为torch张量
            images = torch.from_numpy(image_data)
            # 如果使用CUDA，则将数据移动到GPU
            if self.cuda:
                images = images.cuda()
            # 将图像数据输入到模型中，获取模型的输出
            outputs = self.net(images)
            # 使用解码框工具对模型的输出进行解码
            outputs = self.bbox_util.decode_box(outputs)
            # 对解码后的输出进行非极大值抑制，以去除冗余的检测框
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
            # 如果结果为空，则返回None
            if results[0] is None:
                return None, None, None
            # 从结果中获取目标的标签
            top_label = np.array(results[0][:, 6], dtype='int32')
            # 从结果中获取目标的置信度
            top_conf = results[0][:, 4] * results[0][:, 5]
            # 从结果中获取目标的框的坐标
            top_boxes = results[0][:, :4]
        # 返回目标的标签、框的坐标和置信度
        return top_label, top_boxes, top_conf

    # 计数类别的函数
    # 定义一个名为 count_classes 的方法，接受 self(类实例) 和 top_label(目标标签)两个参数
    def count_classes(self, top_label):
        # 打印目标标签
        print("top_label:", top_label)
        # 创建一个长度为类别数量的零数组，用于存储每个类别的目标数量
        classes_nums = np.zeros([self.num_classes])
        # 遍历每个类别
        for i in range(self.num_classes):
            # 计算当前类别的目标数量，即目标标签等于当前类别的数量
            num = np.sum(top_label == i)
            # 如果当前类别的目标数量大于0，则打印类别名和目标数量
            if num > 0:
                print("class %s: %d" % (self.class_names[i], num))
            # 将当前类别的目标数量存储到数组中
            classes_nums[i] = num
        # 打印每个类别的目标数量
        print("classes_nums:", classes_nums)

    # 裁剪对象的函数
    # 定义一个名为 crop_objects 的方法，接受 self(类实例), image(输入图像), top_boxes(目标框坐标)三个参数
    def crop_objects(self, image, top_boxes):
        # 遍历 top_boxes 中的每一个元素，i 是元素的索引，c 是元素的值
        for i, c in list(enumerate(top_boxes)):
            # 从 top_boxes 中取出第 i 个元素，这个元素是一个包含四个值的列表，分别代表目标框的上、左、下、右四个边的坐标
            top, left, bottom, right = top_boxes[i]
            # 对四个坐标值进行处理，确保它们在图像范围内
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            # 定义裁剪出的图像的保存路径
            dir_save_path = "img_crop"
            # 如果保存路径不存在，则创建该路径
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            # 从输入图像中裁剪出目标对象
            crop_image = image.crop([left, top, right, bottom])
            # 将裁剪出的对象保存为图像文件，文件名为 "crop_" 加上对象的索引，文件格式为 png
            crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
            # 打印保存信息
            print("save crop_" + str(i) + ".png to " + dir_save_path)
