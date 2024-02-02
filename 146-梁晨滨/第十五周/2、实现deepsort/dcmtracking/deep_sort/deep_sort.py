import numpy as np
import torch

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
import time

__all__ = ['DeepSort']


# deepsort主程序
class DeepSort(object):

    def __init__(self, extractor, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, use_cuda=True):

        # 最小置信度，小于这个值，认为是无效物体
        self.min_confidence = min_confidence
        # nms最大重叠比
        self.nms_max_overlap = nms_max_overlap
        # 生成特征向量的方法，可以根据情况替换
        self.extractor = extractor

        max_cosine_distance = max_dist
        # 每个track保留多少历史特征向量，超过这个数，旧的将被淘汰
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    # 根据检测结果更新deepsort状态
    def update(self, bbox_xywh, confidences, ori_img):

        self.height, self.width = ori_img.shape[:2]
        # 在图像按照预测框参数切割出每个预测框特征图
        features = self._get_features(bbox_xywh, ori_img)
        # 将中心点坐标(x, y)转换成左上角坐标(t, l)，wh不动
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        # 根据features和bbox_tlwh生成detections 每个detection有features/tlwh/confidence 三个属性
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if conf > self.min_confidence]

        # 执行nms 去掉重复的detection 其实在目标检测阶段已经做了nms 这里不做也行
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # 卡尔曼滤波:用上一轮结果来计算本轮预测值
        self.tracker.predict()
        # 卡尔曼滤波更新操作
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs

    # xywh转换为中心点(x, y)
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):

        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2]/2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3]/2.
        return bbox_tlwh

    # 将xywh转换为左上角(x1,y1)和右下角坐标(x2, y2)
    def _xywh_to_xyxy(self, bbox_xywh):

        x, y, w, h = bbox_xywh
        x1 = max(int(x-w/2), 0)
        x2 = min(int(x+w/2), self.width - 1)
        y1 = max(int(y-h/2), 0)
        y2 = min(int(y+h/2), self.height - 1)
        return x1, y1, x2, y2

    # 左上角坐标和wh转换为左上角和右下角坐标
    def _tlwh_to_xyxy(self, bbox_tlwh):

        x, y, w, h = bbox_tlwh
        x1, y1 = max(int(x), 0), max(int(y), 0)
        x2, y2 = min(int(x + w), self.width - 1), min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    # 将左上和右下角坐标计算w和h，保留左上角坐标
    def _xyxy_to_tlwh(self, bbox_xyxy):

        x1, y1, x2, y2 = bbox_xyxy
        w, h = int(x2-x1), int(y2-y1)

        return x1, y1, w, h

    # 根据得到的xywh转化为左上和右下角坐标，根据坐标从特征图中切割目标区域特征图
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


