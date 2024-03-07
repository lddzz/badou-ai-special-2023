# 并行调用的线程数
num_parallel_calls = 4
# 输入图像的形状
input_shape = 416
# 最大检测框数
max_boxes = 20
# 亮度抖动的幅度
jitter = 0.3
# 色彩抖动的幅度
hue = 0.1
# 对比度抖动的比例
sat = 1.0
# 亮度抖动的幅度
cont = 0.8
# 轮廓抖动的幅度
bri = 0.1
# 归一化衰减率
norm_decay = 0.99
# 归一化的阈值
norm_epsilon = 1e-3
# 是否使用预训练模型
pre_train = True
# 每个网格预测的锚点数
num_anchors = 9
# 类别数
num_classes = 80
# 是否用于训练
training = True
# 忽略的置信度阈值
ignore_thresh = .5
# 学习率
learning_rate = 0.001
# 训练批次大小
train_batch_size = 10
# 验证批次大小
val_batch_size = 10
# 训练集数量
train_num = 2800
# 验证集数量
val_num = 5000
# 训练轮数
Epoch = 50
# 目标检测中置信度阈值
obj_threshold = 0.5
# 目标检测中非极大值抑制阈值
nms_threshold = 0.5
# GPU索引
gpu_index = "0"
# 日志目录
log_dir = './logs'
# 数据目录
data_dir = './model_data'
# 模型目录
model_dir = './test_model/model.ckpt-192192'
# 是否使用预训练的YOLO3模型
pre_train_yolo3 = True
# YOLO3模型权重路径
yolo3_weights_path = './model_data/yolov3.weights'
# Darknet53模型权重路径
darknet53_weights_path = './model_data/darknet53.weights'
# YOLO锚点路径
anchors_path = './model_data/yolo_anchors.txt'
# 类别列表路径
classes_path = './model_data/coco_classes.txt'
# 图像文件路径
image_file = "./img/img.jpg"
