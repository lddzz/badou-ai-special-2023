# coding=utf-8
import cv2
import time
import torch
from dcmtracking.utils.parser import get_config
from dcmtracking.deep_sort import DeepSort
import numpy as np


class BaseTracker(object):

    def __init__(self):
        """
        deepsort的基类，该类需要被集成并实现detect方法
        Parameters
        ----------
        """
        cfg = get_config()
        cfg.merge_from_file("dcmtracking/deep_sort/deep_sort.yaml")
        self.need_draw_bboxes = cfg.DEEPSORT.NEED_DRAW_BBOXES
        self.need_speed = cfg.DEEPSORT.NEED_SPEED
        self.need_angle = cfg.DEEPSORT.NEED_ANGLE
        self.last_deepsort_outputs = None
        self.frames_count = 0
        self.track_objs = {}
        self.extractor = self.init_extractor()
        self.deepsort = DeepSort(self.extractor,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, # nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        # 记录耗时的dict: det物体检测耗时
        self.cost_dict = {'det': 0, 'deepsort_update': 0 }

    # 初始化特征提取器
    def init_extractor(self):

        raise EOFError("Undefined model type.")

    # 检测，返回目标框参数
    def detect(self):

        raise EOFError("Undefined model type.")

    # 处理视频每一帧
    def deal_one_frame(self, image, speed_skip, need_detect=True):
        self.frames_count += 1
        if self.last_deepsort_outputs is None or need_detect:
            t1 = time.time()
            # 1.用yolo框架先执行目标检测
            _, bboxes = self.detect(image)
            self.cost_dict['det'] += time.time() - t1
            bbox_xywh = []
            confs = []
            bboxes2draw = []
            ids = []
            outputs = []
            if len(bboxes):
                t1 = time.time()
                for x1, y1, x2, y2, _, conf in bboxes:
                    obj = [int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1]
                    bbox_xywh.append(obj)
                    confs.append(conf)

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                # 2.执行deepsort的update
                outputs = self.deepsort.update(xywhs, confss, image)
                self.cost_dict['deepsort_update'] += time.time() - t1
                self.last_deepsort_outputs = outputs
        else:
            outputs = self.last_deepsort_outputs
        for value in list(outputs):
            x1, y1, x2, y2, track_id = value
            track_id = str(track_id)
            ids.append(track_id)
            bboxes.append((x1, y1, x2, y2))
            if track_id in self.track_objs:
                track_obj = self.track_objs[track_id]
            else:
                # 新建一个跟踪对象
                track_obj = {'track_id': track_id, 'location': [], 'center': [], 'speed': 0, 'angle': 0}
            center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
            track_obj['center'].append(center)
            track_obj['location'] = (x1, y1, x2, y2)
            if len(track_obj['center']) < speed_skip:
                speed_frame = 0
            else:
                speed_frame = len(track_obj['center']) - speed_skip
            # 计算速度
            if self.need_speed:
                self.calc_speed(track_obj, speed_frame)
            # 计算运动方向
            if self.need_angle:
                self.calc_angle(track_obj, speed_frame)
            self.track_objs[track_id] = track_obj
            bboxes2draw.append(track_obj)
        if self.need_draw_bboxes:
            image = self.draw_bboxes(image, bboxes2draw)
        return image, ids, bboxes

    # 计算速度
    def calc_speed(self, track_obj, speed_frame):

        speed = euclidean_distance(track_obj['center'][speed_frame], track_obj['center'][-1])
        track_obj['speed'] = speed
        return speed

    # 计算运动方向
    def calc_angle(self, track_obj, speed_frame):

        x1, y1, x2, y2 = [track_obj['center'][speed_frame][0], track_obj['center'][speed_frame][1],
                          track_obj['center'][-1][0], track_obj['center'][-1][1]]
        if x1 == x2:
            return 90
        if y1 == y2:
            return 180
        k = -(y2 - y1) / (x2 - x1)
        # 求反正切，再将得到的弧度转换为度
        result = np.arctan(k) * 57.29577
        # 234象限
        if x1 > x2 and y1 > y2:
            result += 180
        elif x1 > x2 and y1 < y2:
            result += 180
        elif x1 < x2 and y1 < y2:
            result += 360
        # print("直线倾斜角度为：" + str(result) + "度")
        track_obj['angle'] = result
        return result

    # 画框
    def draw_bboxes(self, image, bboxes2draw):

        cls_id = ''
        for track_obj in bboxes2draw:
            x1, y1, x2, y2 = track_obj['location']

            if cls_id == 'eat':
                cls_id = 'eat-drink'
            c1, c2 = (x1, y1), (x2, y2)
            text = track_obj['track_id']
            color = (0, 0, 255)
            if self.need_speed:
                speed = track_obj['speed']

                text += '-' + str(int(speed)) + 'pix/s'
            if self.need_angle:
                angle = track_obj['angle']
                text += '-' + '%.2f' % angle
            cv2.rectangle(image, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(image, text, (c1[0], c1[1] + 10), 0, 1,
                        [0, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        return image


# 计算距离
def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


# 随机0-1概率值
def ran(a=0, b=1):
    return np.random.rand() * (b - a) + a
