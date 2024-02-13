# 导入numpy库，用于进行数值计算
import numpy as np
# 导入torch库，用于进行深度学习计算
import torch
# 从.sort模块中导入Detection类，用于处理检测结果
from .sort.detection import Detection
# 从.sort模块中导入NearestNeighborDistanceMetric类，用于计算最近邻距离
from .sort.nn_matching import NearestNeighborDistanceMetric
# 从.sort模块中导入non_max_suppression函数，用于进行非极大值抑制
from .sort.preprocessing import non_max_suppression
# 从.sort模块中导入Tracker类，用于进行目标跟踪
from .sort.tracker import Tracker


# 定义DeepSort类，用于进行深度排序
class DeepSort(object):
    # 定义初始化函数，输入参数
    # 包括特征提取器: extractor
    # 最大距离: max_dist=0.2
    # 最小置信度: min_confidence=0.3
    # 最大重叠度: nms_max_overlap=1.0
    # 最大IoU距离: max_iou_distance=0.7
    # 最大年龄: max_age=70
    # 初始化次数: n_init=3
    # 是否使用CUDA: use_cuda=True
    def __init__(self, extractor, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, use_cuda=True):
        # 设置最小置信度
        self.min_confidence = min_confidence
        # 设置最大重叠度
        self.nms_max_overlap = nms_max_overlap
        # 设置特征提取器
        self.extractor = extractor
        # 设置最大余弦距离
        max_cosine_distance = max_dist
        # 设置最近邻预算
        nn_budget = 100
        # 创建最近邻距离度量对象
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # 创建跟踪器对象
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    # 定义更新函数，输入参数包括边界框、置信度、原始图像
    def update(self, bbox_xywh, confidences, ori_img):
        # 获取原始图像的高度和宽度
        self.height, self.width = ori_img.shape[:2]
        # 获取特征
        features = self._get_features(bbox_xywh, ori_img)
        # 将边界框从xywh格式转换为tlwh格式
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        # 创建检测对象列表
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]
        # 获取边界框列表: boxes
        boxes = np.array([d.tlwh for d in detections])
        # 获取置信度列表: scores
        scores = np.array([d.confidence for d in detections])
        # 进行非极大值抑制，获取索引
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        # 根据索引获取检测对象列表
        detections = [detections[i] for i in indices]
        # 预测跟踪器的状态
        self.tracker.predict()
        # 更新跟踪器的状态
        self.tracker.update(detections)

        # 初始化输出列表
        outputs = []
        # 遍历跟踪器的轨迹
        for track in self.tracker.tracks:
            # 如果轨迹未被确认或者更新时间超过1，则跳过
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # 获取轨迹的边界框
            box = track.to_tlwh()
            # 将边界框从tlwh格式转换为xyxy格式
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            # 获取轨迹的ID
            track_id = track.track_id
            # 将边界框和轨迹ID添加到输出列表中
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        # 如果输出列表不为空，则将其转换为numpy数组
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        # 返回输出
        return outputs

    # 定义一个静态方法，将边界框从xywh格式转换为tlwh格式
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        # 如果输入的边界框是numpy数组，则复制一份
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        # 如果输入的边界框是torch张量，则克隆一份
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        # 将边界框的x和y坐标转换为左上角的坐标
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        # 返回转换后的边界框
        return bbox_tlwh

    # 定义一个方法，将边界框从xywh格式转换为xyxy格式
    def _xywh_to_xyxy(self, bbox_xywh):
        # 获取边界框的x、y、w、h值
        x, y, w, h = bbox_xywh
        # 计算边界框的左上角和右下角的坐标
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        # 返回转换后的边界框
        return x1, y1, x2, y2

    # 定义一个方法，将边界框从tlwh格式转换为xyxy格式
    def _tlwh_to_xyxy(self, bbox_tlwh):
        # 获取边界框的t、l、w、h值
        x, y, w, h = bbox_tlwh
        # 计算边界框的左上角和右下角的坐标
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        # 返回转换后的边界框
        return x1, y1, x2, y2

    # 定义一个方法，将边界框从xyxy格式转换为tlwh格式
    def _xyxy_to_tlwh(self, bbox_xyxy):
        # 获取边界框的x1、y1、x2、y2值
        x1, y1, x2, y2 = bbox_xyxy
        # 计算边界框的t、l、w、h值
        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        # 返回转换后的边界框
        return t, l, w, h

    # 定义一个方法，获取特征
    def _get_features(self, bbox_xywh, ori_img):
        # 初始化图像裁剪列表
        im_crops = []
        # 遍历每个边界框
        for box in bbox_xywh:
            # 将边界框从xywh格式转换为xyxy格式
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            # 裁剪原始图像
            im = ori_img[y1:y2, x1:x2]
            # 将裁剪后的图像添加到列表中
            im_crops.append(im)
        # 如果图像裁剪列表不为空
        if im_crops:
            # 使用特征提取器提取特征
            features = self.extractor(im_crops)
        else:
            # 否则，特征为空
            features = np.array([])
        # 返回特征
        return features
