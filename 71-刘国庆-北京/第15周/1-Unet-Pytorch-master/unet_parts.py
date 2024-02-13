# 导入PyTorch库，这是一个开源的深度学习平台
import torch
# 导入PyTorch的神经网络库
import torch.nn as nn
# 导入PyTorch的函数库，包含了许多用于构建网络的函数和类
import torch.nn.functional as F


# 定义一个双卷积层类DoubleConv，继承自nn.Module
class DoubleConv(nn.Module):
    # 初始化函数，接收输入通道数为in_channels，输出通道数为out_channels作为参数
    def __init__(self, in_channels, out_channels):
        # 调用父类的初始化函数
        super().__init__()
        # 定义一个双卷积层，包含两个(卷积->批标准化->激活函数)的序列
        self.double_conv = nn.Sequential(
            # 卷积层: 输入通道数为in_channels，输出通道数为out_channels，卷积核大小为3x3，填充为1
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            # 批标准化层: 输入通道数为out_channels
            nn.BatchNorm2d(out_channels),
            # 激活函数层: 使用ReLU激活函数, inplace=True表示直接覆盖原变量
            nn.ReLU(inplace=True),
            # 卷积层: 输入通道数为out_channels，输出通道数为out_channels，卷积核大小为3x3，填充为1
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            # 批标准化层: 输入通道数为out_channels
            nn.BatchNorm2d(out_channels),
            # 激活函数层: 使用ReLU激活函数, inplace=True表示直接覆盖原变量
            nn.ReLU(inplace=True)
        )

    # 前向传播函数，接收一个输入张量x
    def forward(self, x):
        # 将x传入双卷积层并返回结果
        return self.double_conv(x)


# 定义一个下采样层类Down，继承自nn.Module
class Down(nn.Module):
    # 初始化函数，接收输入通道数为in_channels，输出通道数为out_channels作为参数
    def __init__(self, in_channels, out_channels):
        # 调用父类的初始化函数
        super().__init__()
        # 定义一个下采样层，包含一个(最大池化层+双卷积层)的序列
        self.maxpool_conv = nn.Sequential(
            # 最大池化层，池化窗口为2x2
            nn.MaxPool2d(2),
            # 双卷积层: 输入通道数为in_channels，输出通道数为out_channels
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    # 前向传播函数，接收一个输入张量x
    def forward(self, x):
        # 将x传入下采样层并返回结果
        return self.maxpool_conv(x)


# 定义一个上采样层类Up，继承自nn.Module
class Up(nn.Module):
    # 初始化函数，接收输入通道数in_channels、输出通道数out_channels和是否使用双线性插值bilinear=True作为参数
    def __init__(self, in_channels, out_channels, bilinear=True):
        # 调用父类的初始化函数
        super().__init__()
        # 根据是否使用双线性插值来选择上采样的方式
        if bilinear:
            # 使用双线性插值进行上采样: 缩放因子为2，模式为双线性插值，对齐方式为True
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 否则
        else:
            # 使用转置卷积进行上采样: 输入通道数为in_channels // 2，输出通道数为in_channels//2，卷积核大小为2x2，步长为2
            self.up = nn.ConvTranspose2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=2,
                                         stride=2)
        # 双卷积层: 输入通道数为in_channels，输出通道数为out_channels
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    # 前向传播函数，接收两个输入张量x1和x2
    def forward(self, x1, x2):
        # 对x1进行上采样
        x1 = self.up(x1)
        # 计算x2和上采样后的x1在高和宽上的差异: 高度差diffY和宽度差diffX
        #  x2.size()[2]和x1.size()[2]分别表示x2和x1的高度
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        #  x2.size()[3]和x1.size()[3]分别表示x2和x1的宽度
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        # 对上采样后的x1进行填充，使其和x2在高和宽上一致
        x1 = F.pad(
            x1,
            [
                # 左填充: 高度差的一半，向下取整
                torch.div(diffX, 2, rounding_mode='trunc'),
                # 右填充: 高度差的一半减去高度差，向下取整
                torch.div(diffX, 2, rounding_mode='trunc') - diffX,
                # 上填充: 宽度差的一半，向下取整
                torch.div(diffY, 2, rounding_mode='trunc'),
                # 下填充: 宽度差的一半减去宽度差，向下取整
                torch.div(diffY, 2, rounding_mode='trunc') - diffY
            ]
        )
        # 将x2和填充后的x1在通道维度上进行拼接
        x = torch.cat([x2, x1], dim=1)
        # 将拼接后的x传入双卷积层并返回结果
        return self.conv(x)


# 定义一个输出层类OutConv，继承自nn.Module
class OutConv(nn.Module):
    # 初始化函数，接收输入通道数和输出通道数作为参数
    def __init__(self, in_channels, out_channels):
        # 调用父类的初始化函数
        super().__init__()
        # 定义一个卷积层，卷积核大小为1x1, 输入通道数为in_channels，输出通道数为out_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    # 前向传播函数，接收一个输入张量x
    def forward(self, x):
        # 将x传入卷积层并返回结果
        return self.conv(x)
