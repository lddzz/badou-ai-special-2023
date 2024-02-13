# 导入time库，用于计算时间
import time
# 导入OpenCV库，用于图像处理
import cv2
# 导入NumPy库，用于进行数值计算
import numpy as np
# 导入PyTorch库，用于深度学习模型的训练和推理
import torch
# 从dcmtracking.deep_sort模块导入DeepSort类，用于实现DeepSort算法
from dcmtracking.deep_sort import DeepSort
# 从dcmtracking.utils.parser模块导入get_config函数，用于获取配置信息
from dcmtracking.utils.parser import get_config


# 定义一个名为BaseTracker的类，作为所有跟踪器的基类
class BaseTracker(object):
    # 定义类的初始化方法
    def __init__(self):
        # 获取配置信息
        config = get_config()
        # 从指定的yaml文件中加载并合并配置信息
        config.merge_from_file("dcmtracking/deep_sort/deep_sort.yaml")
        # 是否需要绘制边界框
        self.need_draw_bboxes = config.DEEPSORT.NEED_DRAW_BBOXES
        # 是否需要计算速度
        self.need_speed = config.DEEPSORT.NEED_SPEED
        # 是否需要计算角度
        self.need_angle = config.DEEPSORT.NEED_ANGLE
        # 存储上一次DeepSort的输出结果
        self.last_deepsort_outputs = None
        # 记录处理的帧数frames_count
        self.frames_count = 0
        # 存储跟踪的目标对象track_objs
        self.track_objs = {}
        # 初始化特征提取器extractor
        self.extractor = self.init_extractor()

        # 初始化DeepSort对象，DeepSort是一个用于目标跟踪的算法
        # extractor: 特征提取器对象，用于从图像中提取特征
        # max_dist: 两个目标之间的最大距离阈值
        # min_confidence: 目标的最小置信度阈值
        # nms_max_overlap: 执行非最大抑制（NMS）时，两个边界框的最大重叠比例阈值
        # max_iou_distance: 执行IOU匹配时，两个边界框的最大IOU距离阈值
        # max_age: 删除一个跟踪目标之前，可以容忍的最大连续未检测到该目标的帧数阈值
        # n_init: 一个跟踪目标被确认之前，需要连续检测到该目标的帧数阈值
        # use_cuda: 是否使用CUDA进行计算，如果为True，将使用GPU进行计算，否则将使用CPU进行计算
        self.deepsort = DeepSort(
            extractor=self.extractor,
            max_dist=config.DEEPSORT.MAX_DIST,
            min_confidence=config.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=config.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=config.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=config.DEEPSORT.MAX_AGE,
            n_init=config.DEEPSORT.N_INIT,
            use_cuda=True
        )
        # 记录检测和DeepSort更新的时间消耗
        # 初始化一个字典用于记录目标检测和DeepSort更新的时间消耗
        # 'det': 记录目标检测的时间消耗
        # 'deepsort_update': 记录DeepSort更新的时间消耗
        self.cost_dict = {"det": 0, "deepsort_update": 0}

    # 定义初始化特征提取器的方法，具体实现由子类完成
    def init_extractor(self):
        raise EOFError("未定义模型类型")

    # 定义检测方法，具体实现由子类完成
    def detect(self):
        raise EOFError("未定义模型类型")

    # 定义处理一帧图像的方法:deal_one_frame
    # image:可能是一帧视频或者一个静态图像，将在这个方法中进行处理
    # speed_skip:表示在计算目标速度时需要跳过的帧数
    # need_detect:表示是否需要进行目标检测
    # 返回处理后的图像、检测到的目标的ID列表和对应的边界框列表
    def deal_one_frame(self, image, speed_skip, need_detect=True):
        # Step 1: 帧数加一
        self.frames_count += 1
        # Step 2:如果没有上一次的DeepSort输出结果或者需要进行检测
        if self.last_deepsort_outputs is None or need_detect:
            # Step 2.1: 记录开始时间
            start_time = time.time()
            # Step 2.2: 进行检测，获取边界框
            _, bboxes = self.detect(image)
            # Step 2.3: 计算检测的时间消耗
            self.cost_dict['det'] += time.time() - start_time
            # Step 2.4: 初始化一些变量
            # bbox_xywh:存储边界框的中心坐标和宽高
            # confs:置信度
            # bboxes2draw:需要绘制的边界框
            # ids:目标的ID
            # outputs:DeepSort的输出结果的列表
            bbox_xywh = []
            confs = []
            bboxes2draw = []
            ids = []
            outputs = []
            # Step 2.5: 如果检测到了目标
            if len(bboxes):
                # Step 2.5.1: 记录开始时间
                start_time = time.time()
                # Step 2.5.2: 遍历每一个边界框
                for x1, y1, x2, y2, _, conf in bboxes:
                    # 计算边界框的中心坐标和宽高
                    obj = [
                        int((x1 + x2) / 2),
                        int((y1 + y2) / 2),
                        x2 - x1,
                        y2 - y1
                    ]
                    # 将结果添加到存储边界框列表中
                    bbox_xywh.append(obj)
                    # 将置信度添加到列表中
                    confs.append(conf)
                # Step 2.5.3: 将存储边界框列表、置信度列表转换为张量
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                # Step 2.5.4: 使用DeepSort进行更新存储边界框列表、置信度列表、图像为输出结果outputs
                outputs = self.deepsort.update(xywhs, confss, image)
                # Step 2.5.5: 计算DeepSort更新的时间消耗
                self.cost_dict['deepsort_update'] += time.time() - start_time
                # Step 2.5.6: 存储DeepSort的输出结果
                self.last_deepsort_outputs = outputs
        # Step 3: 如果不需要进行检测，则直接使用上一次的DeepSort输出结果
        else:
            outputs = self.last_deepsort_outputs
        # Step 4: 遍历DeepSort的输出结果
        for output in list(outputs):
            # Step 4.1: 获取边界框的坐标和目标的ID
            x1, y1, x2, y2, track_id = output
            # 将目标的ID转换为字符串
            track_id = str(track_id)
            # 将目标的ID添加到列表中
            ids.append(track_id)
            # 将边界框的坐标添加到列表中
            bboxes.append((x1, y1, x2, y2))
            # Step 4.2: 检查目标(ID)是否已经存在于跟踪对象中
            # 检查当前目标（由track_id标识）是否已经存在于跟踪对象self.track_objs中
            if track_id in self.track_objs:
                # 如果目标已经存在，那么直接获取该目标的信息
                track_obj = self.track_objs[track_id]
                # 如果目标不存在，那么在后续的代码中创建新的跟踪对象
            else:
                # 创建新的跟踪对象
                # 创建一个名为track_obj的字典，用于存储一个跟踪目标的信息
                # 'track_id':目标的ID
                # 'location':目标的位置信息
                # 'center':目标的中心坐标
                # 'speed':目标的速度
                # 'angle':目标的运动方向
                track_obj = {
                    'track_id': track_id,
                    'location': [],
                    'center': [],
                    'speed': 0,
                    'angle': 0
                }
            # Step 4.3: 计算边界框的中心坐标
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            # 将中心坐标添加到跟踪对象中
            track_obj['center'].append(center)
            # 更新跟踪对象的位置
            track_obj['location'] = (x1, y1, x2, y2)
            # Step 4.4: 检查跟踪对象的中心坐标数量是否小于速度跳过的帧数
            # 检查目标被跟踪的帧数是否小于需要跳过的帧数
            if len(track_obj['center']) < speed_skip:
                # 如果目标被跟踪的帧数小于需要跳过的帧数，那么从第一帧开始计算速度
                speed_frame = 0
            else:
                # 如果目标被跟踪的帧数大于或等于需要跳过的帧数，那么从跳过指定数量的帧后开始计算速度
                speed_frame = len(track_obj['center']) - speed_skip
            # Step 4.5: 如果需要，计算速度
            if self.need_speed:
                # 调用calc_speed方法计算速度
                self.calc_speed(track_obj, speed_frame)
            # Step 4.6: 如果需要，计算运动方向
            if self.need_angle:
                # 调用calc_angle方法计算运动方向
                self.calc_angle(track_obj, speed_frame)
            # Step 4.7: 更新跟踪对象
            self.track_objs[track_id] = track_obj
            # Step 4.8: 将跟踪对象添加到需要绘制的边界框列表中
            bboxes2draw.append(track_obj)
        # Step 5: 如果需要绘制边界框
        if self.need_draw_bboxes:
            # 调用draw_bboxes方法绘制边界框
            image = self.draw_bboxes(image, bboxes2draw)
        # Step 6: 返回图像image、目标的IDids和边界框的坐标bboxes
        return image, ids, bboxes

    # 定义计算速度的方法
    def calc_speed(self, track_obj, speed_frame):
        # 计算速度
        speed = euclidean_distance(track_obj['center'][speed_frame], track_obj['center'][-1])
        # 更新跟踪对象的速度
        track_obj['speed'] = speed
        # 返回速度
        return speed

    # 定义计算角度的方法
    def calc_angle(self, track_obj, speed_frame):
        # 获取两个中心坐标
        x1, y1, x2, y2 = [
            track_obj['center'][speed_frame][0],
            track_obj['center'][speed_frame][1],
            track_obj['center'][-1][0],
            track_obj['center'][-1][1]
        ]
        # 如果x坐标相等
        if x1 == x2:
            return 90
        # 如果y坐标相等
        if y1 == y2:
            return 180
        # 计算斜率
        k = -(y2 - y1) / (x2 - x1)
        # 计算角度
        result = np.arctan(k) * 57.29577
        # 如果第一个中心坐标在第二个中心坐标的右上方
        if x1 > x2 and y1 > y2:
            # 角度加180度
            result += 180
        # 如果第一个中心坐标在第二个中心坐标的右下方
        elif x1 > x2 and y1 < y2:
            # 角度加180度
            result += 180
        # 如果第一个中心坐标在第二个中心坐标的左下方
        elif x1 < x2 and y1 < y2:
            # 角度加360度
            result += 360
        # 更新跟踪对象的角度
        track_obj['angle'] = result
        # 返回角度
        return result

    # 定义绘制边界框的方法
    def draw_bboxes(self, image, bboxes2draw):
        # 初始化类别ID
        cls_id = ''
        # 遍历需要绘制的边界框
        for track_obj in bboxes2draw:
            # 获取边界框的坐标
            x1, y1, x2, y2 = track_obj['location']
            # 如果类别ID为'eat'
            if cls_id == 'eat':
                # 将类别ID改为eat-drink
                cls_id = 'eat-drink'
            # 获取边界框的两个对角坐标
            c1, c2 = (x1, y1), (x2, y2)
            # 获取目标的ID
            text = track_obj['track_id']
            # 设置颜色为红色
            color = (0, 0, 255)
            # 如果需要显示速度
            if self.need_speed:
                # 获取速度
                speed = track_obj['speed']
                # 将速度添加到文本中
                text += '-' + str(int(speed)) + 'pix/s'
            # 如果需要显示角度
            if self.need_angle:
                # 获取角度
                angle = track_obj['angle']
                # 将角度添加到文本中
                text += '-' + '%.2f' % angle
            # 在图像上绘制边界框
            cv2.rectangle(image, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
            # 在图像上绘制文本
            cv2.putText(image, text, (c1[0], c1[1] + 10), 0, 1,
                        [0, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        # 返回图像
        return image


# 定义计算欧氏距离的函数
def euclidean_distance(p1, p2):
    # 计算并返回欧氏距离
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# 定义生成随机数的函数
def ran(a=0, b=1):
    # 生成并返回一个在[a, b]范围内的随机数
    return np.random.rand() * (b - a) + a
