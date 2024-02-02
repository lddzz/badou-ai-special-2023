import sys
from dcmtracking.utils.parser import get_config
from dcmtracking.deep_sort import DeepSort
import torch
import cv2
import time
import os
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
DEEP_SORT_PATH = FILE_PATH.parents[1]

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file(os.path.join(DEEP_SORT_PATH, "deep_sort.yaml"))
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

point_fps = 48
save_dict = {'cur_num': 0}
ACCIENT_SURE = 2
ACCIENT_WARMING = 1
ACCIENT_NORMAL = 0

def plot_bboxes(image, bboxes, line_thickness=None):
    cls_id = ''
    for car in bboxes:
        x1, y1, x2, y2 = car['location']
        lastdistence = car['lastdistence']
        points = car['points']
        track_id = car['track_id']
        speed = car['speed']
        stop = car['stop']
        accident = car['accident']
        angle = car['angle']
        speed_a = car['speed_a']
        angle_a = car['angle_a']
        if accident == ACCIENT_SURE:
            color = (0, 0, 255)
        elif accident == ACCIENT_WARMING:

            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'eat':
            cls_id = 'eat-drink'
        c1, c2 = (x1, y1), (x2, y2)
        save_dict['cur_num'] += 1


        if True:
            if int(speed) > 5:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(image, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)

            cv2.putText(image, '{}'.format(int(speed)), (c1[0], c1[1] + 10), 0, 1,
                        [0, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        if len(points) < point_fps:
            start = 0
        else:
            start = len(points) - point_fps
        last_point = points[start]
        for point in points[start:-1]:

            last_point = point

    return image

def update_tracker(target_detector, image, cars, speed_skip, stop_speed=1, stop_round=24, max_speed=10,
                   speed_a_times=10, min_speed_a=-3, context=None, last_bboxes=None, last_outputs=None, tt_dict=None):


        t1 = time.time()
        if last_bboxes is None:
            _, bboxes = target_detector.detect(image)
        else:
            bboxes = last_bboxes
        tt_dict['1'] += time.time() - t1

        bbox_xywh = []
        confs = []
        bboxes2draw = []

        ids = []
        outputs = None
        if len(bboxes):

            t1 = time.time()
            for x1, y1, x2, y2, _, conf in bboxes:
                
                obj = [
                    int((x1+x2)/2), int((y1+y2)/2),
                    x2-x1, y2-y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)

            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)


            if last_outputs is None:
                outputs = deepsort.update(xywhs, confss, image, tt_dict)
            else:
                outputs = last_outputs
            tt_dict['2'] += time.time() - t1
            t1 = time.time()
            for value in list(outputs):
                x1,y1,x2,y2,track_id = value
                track_id = str(track_id)
                ids.append(track_id)

                if track_id in cars:
                    car = cars[track_id]
                else:

                    car = {'track_id': track_id, 'distances': [], 'points': [], 'lastpoint': None, 'speed': 0, 'speed_a': 0,
                           'max_speed': 0, 'last_not_stop_index': -1, 'last_frame_stop': False, 'stop': False, 'accident': ACCIENT_NORMAL, 'speeds': [], 'speeds_a': [], 'angles': [], 'angles_a': [],
                           'stop_count': 0, 'stop_calc': []}

                center = (int((x2 + x1) / 2), int((y2 + y1)/2))
                car['points'].append(center)
                lastpoint = car['lastpoint']
                lastdistence = 0
                if lastpoint is not None:
                    lastdistence = euclidean_distance(center, lastpoint)
                    car['distances'].append(lastdistence)
                car['lastdistence'] = lastdistence
                car['lastpoint'] = (center)
                car['location'] = (x1,y1,x2,y2)

                if len(car['points']) < speed_skip:
                    speed_frame = 0
                else:
                    speed_frame = len(car['points']) - speed_skip

                speed = euclidean_distance(car['points'][speed_frame], car['points'][-1]) * 1000 / (((x2 - x1)*(y2 - y1))**0.5) / 12
                if speed > 10:
                    if ran() > 0.7:
                        speed = ran(car['speed'] - 1, car['speed'] + 1)
                        speed = 10 if speed > 10 else speed
                        speed = ran(2, 5) if speed < 0 else speed
                    else:
                        speed = car['speed']

                angle = calc_angle(car['points'][speed_frame][0], car['points'][speed_frame][1], car['points'][-1][0], car['points'][-1][1])
                car['speed'] = speed
                car['speeds'].append(int(speed))
                car['angle'] = angle
                car['angles'].append(int(angle))
                speed_a = car['speeds'][-1] - car['speeds'][speed_frame]
                car['speed_a'] = speed_a
                car['speeds_a'].append(int(speed_a))
                angle_a = abs(car['angles'][-1] - car['angles'][speed_frame])
                car['angle_a'] = angle_a
                car['angles_a'].append(int(angle_a))
                if speed > car['max_speed']:
                    car['max_speed'] = speed
                if car['stop']:
                    if speed > stop_speed:
                        car['stop'] = False

                        car['stop_calc'][-1][4] = car['stop_count']
                        print("car['stop_count']:", car['stop_count'], speed, car['points'][speed_frame], car['points'][-1], x2 - x1, y2 - y1)
                        car['stop_count'] = 0
                        car['last_not_stop_index'] = len(car['speeds_a'])-1
                    else:
                        car['stop'] = True
                        car['stop_count'] += 1

                        if context['stop_mean'] != 0 and context['stop_variance'] != 0:
                            if car['stop_count'] > context['stop_mean'] + 3 * context['stop_variance']:
                                print(
                                    "ACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SURE")
                                car['accident'] = ACCIENT_SURE
                            elif car['stop_count'] > context['stop_mean'] + 2 * context['stop_variance']:
                                car['accident'] = ACCIENT_WARMING
                                print(
                                    "ACCIENT_WARMINGACCIENT_WARMINGACCIENT_WARMINGACCIENT_WARMINGACCIENT_WARMING")
                else:

                    if speed > stop_speed:
                        car['stop'] = False
                        car['stop_count'] = 0
                        car['last_not_stop_index'] = len(car['speeds_a'])-1
                    else:
                        if car['max_speed'] > max_speed:
                            car['stop_calc'].append([x1, y1 ,x2 ,y2 ,1])
                            car['stop'] = True
                            car['stop_count'] = 1
                        else:
                            car['stop_count'] = 0
                cars[track_id] = car

                bboxes2draw.append(car)


        tt_dict['3'] += time.time() - t1

        image = plot_bboxes(image, bboxes2draw)

        return image, ids, bboxes, cars, outputs

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

def calc_angle(x1, y1, x2, y2):
    if x1 == x2:
        return 90
    if y1 == y2:
        return 180
    k = -(y2 - y1) / (x2 - x1)

    result = np.arctan(k) * 57.29577

    if x1 > x2 and y1 > y2:
        result += 180
    elif x1 > x2 and y1 < y2:
        result += 180
    elif x1 < x2 and y1 < y2:
        result += 360

    return result

def ran(a=0, b=1):
    return np.random.rand() * (b - a) + a