# 导入logging库，用于记录日志
import logging
# 导入OpenCV库，用于图像和视频处理
import cv2
# 导入NumPy库，用于数组和矩阵运算
import numpy as np
# 导入PyTorch库，用于深度学习模型的训练和推理
import torch
# 导入torchvision库的transforms模块，用于图像预处理
import torchvision.transforms as transforms
# 从当前目录的model模块中导入Net类，用于创建深度学习模型
from .model import Net


# 定义Extractor类，继承 object 类，用于特征提取
class Extractor(object):
    # 初始化函数，接收两个参数，分别表示模型路径model_path和是否使用CUDAuse_cuda=True
    def __init__(self, model_path, use_cuda=True):
        # 创建Net类的实例，用于特征提取, reid=True表示使用ReID模型
        self.net = Net(reid=True)
        # 判断是否使用CUDA，如果CUDA可用并且use_cuda为True，则使用CUDA，否则使用CPU
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        # 加载模型权重:
        # 使用torch.load加载模型权重，
        # map_location参数指定将所有张量都映射到CPU上
        # storage 是一个包含了张量数据的存储对象，loc 是存储对象的位置。
        # 从加载的字典中获取键为'net_dict'的项，这通常是模型的state_dict（一个包含了模型所有权重的字典）
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        # 调用load_state_dict将加载的权重赋值给模型
        self.net.load_state_dict(state_dict)
        # 调用getLogger创建日志记录器root.tracker
        logger = logging.getLogger("root.tracker")
        # 记录日志信息
        logger.info(f"加载权重从{model_path}...完成！")
        # 将模型移动到指定的设备（CPU或GPU）
        self.net.to(self.device)
        # 定义图像的大小(64, 128)
        self.size = (64, 128)
        # 定义图像的预处理操作，包括转换为Tensor和标准化
        # 将图像转换为Tensor，并且自动将数据范围归一化到0-1
        # 对图像进行标准化，[0.485, 0.456, 0.406]是RGB三个通道的均值，[0.229, 0.224, 0.225]是RGB三个通道的标准差
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # 定义预处理函数preprocess，接收一个参数image_crops，表示图像的裁剪区域
    def preprocess(self, image_crops):
        # 定义内部函数，用于调整图像的大小并归一化
        def resize(image, size):
            # 将图像的数据类型转换为float32，并将像素值归一化到0-1的范围
            # 使用OpenCV的resize函数将图像的大小调整为指定的size
            return cv2.resize(image.astype(np.float32) / 255., size)

        # 对每个裁剪区域进行预处理，并将结果合并为一个批次
        # 对image_crops中的每个图像进行预处理，包括调整大小、归一化、转换为Tensor、标准化，并增加一个批次维
        # 使用torch.cat将所有处理后的图像沿着批次维拼接起来，形成一个批次的图像
        # 将结果转换为float类型
        image_batch = torch.cat([self.norm(resize(image, self.size)).unsqueeze(0) for image in image_crops],
                                dim=0).float()
        # 返回预处理后的图像批次
        return image_batch

    # 定义调用函数，接收一个参数，表示图像的裁剪区域
    def __call__(self, image_crops):
        # 对裁剪区域进行预处理,生成图像批次image_batch
        image_batch = self.preprocess(image_crops)
        # 不计算梯度，以节省内存并加速计算
        with torch.no_grad():
            # 将图像批次移动到指定的设备（CPU或GPU）
            image_batch = image_batch.to(self.device)
            # 使用模型提取特征features
            features = self.net(image_batch)
        # 将特征从GPU移动到CPU，并转换为NumPy数组
        return features.cpu().numpy()


# 如果当前脚本被作为主程序运行
if __name__ == '__main__':
    # 读取图像，并将颜色空间从BGR转换为RGB
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    # 创建Extractor类的实例
    extr = Extractor("checkpoint/ckpt.t7")
    # 使用Extractor提取图像的特征
    feature = extr(img)
    # 打印特征的形状
    print(feature.shape)
