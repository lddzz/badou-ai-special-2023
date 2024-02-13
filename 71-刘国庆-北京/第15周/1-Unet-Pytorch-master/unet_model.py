# 导入unet_parts模块中的所有内容，这个模块包含了U-Net模型的各个组成部分
from .unet_parts import *


# 定义U-Net模型类，继承自nn.Module
class UNet(nn.Module):
    # 初始化函数，输入参数包括输入通道数n_channels、输出类别数n_classes和是否使用双线性插值上采样bilinear=True
    def __init__(self, n_channels, n_classes, bilinear=True):
        # 调用父类的初始化函数
        super().__init__()
        # 保存输入通道数n_channels
        self.n_channels = n_channels
        # 保存输出类别数n_classes
        self.n_classes = n_classes
        # 保存是否使用双线性插值上采样的标志bilinear
        self.bilinear = bilinear

        # 定义U-Net的各个组成部分，包括初始卷积层、四个下采样层、四个上采样层和输出卷积层
        # 创建一个DoubleConv对象，输入通道数为n_channels，输出通道数为64。
        # DoubleConv是一个包含两个卷积层的模块，每个卷积层后都有一个批量归一化层和ReLU激活函数。
        self.inc = DoubleConv(n_channels, 64)

        # Down模块是一个包含一个最大池化层和一个DoubleConv模块的下采样模块。
        # 创建一个Down对象，输入通道数为64，输出通道数为128。
        self.down1 = Down(64, 128)
        # 创建一个Down对象，输入通道数为128，输出通道数为256。
        self.down2 = Down(128, 256)
        # 创建一个Down对象，输入通道数为256，输出通道数为512。
        self.down3 = Down(256, 512)
        # 创建一个Down对象，输入通道数为512，输出通道数为512。
        self.down4 = Down(512, 512)

        # Up模块是一个包含一个上采样操作和一个DoubleConv模块的上采样模块。
        # 创建一个Up对象，输入通道数为1024，输出通道数为256，是否使用双线性插值由bilinear参数决定。
        self.up1 = Up(1024, 256, bilinear)
        # 创建一个Up对象，输入通道数为512，输出通道数为128，是否使用双线性插值由bilinear参数决定。
        self.up2 = Up(512, 128, bilinear)
        # 创建一个Up对象，输入通道数为256，输出通道数为64，是否使用双线性插值由bilinear参数决定。
        self.up3 = Up(256, 64, bilinear)
        # 创建一个Up对象，输入通道数为128，输出通道数为64，是否使用双线性插值由bilinear参数决定。
        self.up4 = Up(128, 64, bilinear)

        # OutConv模块是一个包含一个卷积层的模块，用于生成最终的输出。
        # 创建一个OutConv对象，输入通道数为64，输出通道数为n_classes。
        self.outc = OutConv(64, n_classes)

    # 定义前向传播函数
    def forward(self, x):
        # 通过U-Net的各个组成部分进行前向传播
        # 将输入x通过初始的DoubleConv模块进行处理，得到x1
        x1 = self.inc(x)
        # 将x1通过第一个Down模块进行处理，进行下采样并通过两次卷积，得到x2
        x2 = self.down1(x1)
        # 将x2通过第二个Down模块进行处理，进行下采样并通过两次卷积，得到x3
        x3 = self.down2(x2)
        # 将x3通过第三个Down模块进行处理，进行下采样并通过两次卷积，得到x4
        x4 = self.down3(x3)
        # 将x4通过第四个Down模块进行处理，进行下采样并通过两次卷积，得到x5
        x5 = self.down4(x4)

        # 将x5和x4通过第一个Up模块进行处理，进行上采样并通过两次卷积，得到新的x
        x = self.up1(x5, x4)
        # 将新的x和x3通过第二个Up模块进行处理，进行上采样并通过两次卷积，得到更新的x
        x = self.up2(x, x3)
        # 将更新的x和x2通过第三个Up模块进行处理，进行上采样并通过两次卷积，得到再次更新的x
        x = self.up3(x, x2)
        # 将再次更新的x和x1通过第四个Up模块进行处理，进行上采样并通过两次卷积，得到最终的x
        x = self.up4(x, x1)

        # 最后通过输出卷积层得到输出结果
        result = self.outc(x)
        # 返回输出结果
        return result


# 如果当前脚本被直接运行，而不是被导入
if __name__ == '__main__':
    # 创建一个U-Net模型实例，输入通道数为3，输出类别数为1
    net = UNet(n_channels=3, n_classes=1)
    # 打印模型的结构
    print(f"模型的结构{net}")
