# coding=utf-8

from dcmtracking.deep_sort.tracker.base_tracker import BaseTracker
from dcmtracking.detection.yolo_fastestv2.yolo_fastestv2 import YOLO
from dcmtracking.deep_sort.deep.feature_extractor import Extractor
import torch


class YoloFastestV2DeepSortTracker(BaseTracker):

    def __init__(self, need_speed=False, need_angle=False):
        BaseTracker.__init__(self)
        self.yolo = YOLO()

    def init_extractor(self):
        model_path = "dcmtracking/deep_sort/deep/checkpoint/ckpt.t7"
        return Extractor(model_path, use_cuda=torch.cuda.is_available())

    def detect(self, im):
        pred_boxes = self.yolo.detect_image(im)
        results = []
        for pred_box in pred_boxes:
            lbl = pred_box[4]
            if lbl == 0:
                results.append(pred_box)
        return im, results
