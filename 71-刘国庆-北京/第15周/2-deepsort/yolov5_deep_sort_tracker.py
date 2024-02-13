# 导入OpenCV库，用于图像和视频处理
import cv2
# 导入PyTorch库，用于深度学习模型的训练和推理
import torch
# 导入PIL库，用于图像处理
from PIL import Image
# 导入特征提取器
from dcmtracking.deep_sort.deep.feature_extractor import Extractor
# 导入基础追踪器
from dcmtracking.deep_sort.tracker.base_tracker import BaseTracker
# 导入YOLO目标检测模型
from dcmtracking.detection.yolov5.yolo import YOLO


# 定义Yolov5DeepSortTracker类，继承自BaseTracker
class Yolov5DeepSortTracker(BaseTracker):
    # 初始化函数，接收两个参数，分别表示是否需要计算速度need_speed=False和角度need_angle=False
    def __init__(self, need_speed=False, need_angle=False):
        # 调用父类的初始化函数
        super().__init__()
        # 创建YOLO目标检测模型的实例
        self.yolo = YOLO()

    # 初始化特征提取器
    def init_extractor(self):
        # 定义模型路径
        model_path = "dcmtracking/deep_sort/deep/checkpoint/ckpt.t7"
        # 创建特征提取器的实例，并返回
        return Extractor(model_path, use_cuda=torch.cuda.is_available())

    # 定义目标检测函数，接收一个参数，表示输入的图像
    def detect(self, image):
        # 获取图像的高度、宽度和通道数
        h, w, _ = image.shape
        # 将图像从BGR格式转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 将NumPy数组转换为PIL图像
        image_pil = Image.fromarray(image)
        # 初始化预测的边界框列表
        pred_boxes = []
        # 使用YOLO模型进行目标检测，获取标签、边界框和置信度
        top_label, top_boxes, top_conf = self.yolo.detect_image(image_pil)
        # 如果检测到的标签不为空
        if top_label is not None:
            # 遍历每一个边界框、标签和置信度
            for (y1, x1, y2, x2), label, conf in zip(top_boxes, top_label, top_conf):
                # 如果标签不等于0，跳过当前循环
                if label != 0:
                    continue
                # 将边界框、标签和置信度添加到预测的边界框列表中
                pred_boxes.append((int(x1), int(y1), int(x2), int(y2), label, conf))
        # 返回图像和预测的边界框列表
        return image, pred_boxes
