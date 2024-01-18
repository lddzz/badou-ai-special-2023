# 定义一个名为Config的类
class Config:
    # 类的初始化函数，用于创建类的实例时自动执行
    def __init__(self):
        # 定义锚框的尺度，用于目标检测
        self.anchor_box_scales = [128, 256, 512]
        # 定义锚框的宽高比
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        # 定义RPN网络中的stride值
        self.rpn_stride = 16
        # 定义感兴趣区域(Region of Interest)的数量
        self.num_rois = 32
        # 定义一个布尔值，控制输出的详细程度
        self.verbose = True
        # 定义模型文件的存储路径
        self.model_path = "logs/model.h5"
        # 定义RPN中锚框的最小重叠比例
        self.rpn_min_overlap = 0.3
        # 定义RPN中锚框的最大重叠比例
        self.rpn_max_overlap = 0.7
        # 定义分类器中的最小重叠比例
        self.classifier_min_overlap = 0.1
        # 定义分类器中的最大重叠比例
        self.classifier_max_overlap = 0.5
        # 定义分类器回归的标准化参数
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
