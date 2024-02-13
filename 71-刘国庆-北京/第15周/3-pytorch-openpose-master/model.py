# 导入Python的collections模块中的OrderedDict类，
# OrderedDict是一个字典子类，它记住了元素插入的顺序
from collections import OrderedDict
# 导入torch库，torch是一个开源的机器学习库，提供了广泛的模块和类，如张量操作，自动求导等
import torch
# 从torch.nn模块导入nn，nn模块包含了各种用于构建神经网络的类和函数
import torch.nn as nn


# 定义一个名为make_layers的函数，接受两个参数：block和no_relu_layers
def make_layers(block, no_relu_layers):
    # 初始化一个空列表，用于存储层
    layers = []
    # 遍历block中的每一项，每一项由层的名称和一个列表v组成，列表v包含了该层的参数
    for layer_name, v in block.items():
        # 判断层的名称中是否包含'pool'，如果包含，说明这是一个池化层
        if 'pool' in layer_name:
            # 创建一个最大池化层，核大小、步长和填充值由v中的元素提供
            layer = nn.MaxPool2d(
                kernel_size=v[0],
                stride=v[1],
                padding=v[2]
            )
            # 将创建的池化层添加到layers列表中，每个元素是一个元组，包含层的名称和层对象
            layers.append((layer_name, layer))
        # 如果层的名称中不包含'pool'，说明这是一个卷积层
        else:
            # 创建一个二维卷积层，输入通道数、输出通道数、核大小、步长和填充值由v中的元素提供
            conv2d = nn.Conv2d(
                in_channels=v[0],
                out_channels=v[1],
                kernel_size=v[2],
                stride=v[3],
                padding=v[4]
            )
            # 将创建的卷积层添加到layers列表中
            layers.append((layer_name, conv2d))
            # 如果层的名称不在no_relu_layers列表中，说明需要在这个卷积层后添加一个ReLU激活函数
            if layer_name not in no_relu_layers:
                # 创建一个ReLU激活函数层，参数inplace=True表示直接修改输入，不需要额外的空间
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
    # 使用layers列表中的层创建一个顺序模型，layers列表中的每个元素都会按照顺序添加到模型中
    # 使用OrderedDict可以保证层的顺序不会改变
    return nn.Sequential(OrderedDict(layers))


# 定义一个名为bodypose_model的类，该类继承自nn.Module，nn.Module是所有神经网络模块的基类
class bodypose_model(nn.Module):
    # 定义初始化函数，接受self参数，self代表类的实例
    def __init__(self):
        # 调用父类的初始化函数
        super(bodypose_model, self).__init__()
        # 定义一个列表，包含不需要ReLU激活函数的层的名称
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        # 初始化一个空字典，用于存储各个块的信息
        blocks = {}
        # 定义第一个块的信息，使用OrderedDict保证元素的插入顺序
        # 创建一个有序字典block0
        block0 = OrderedDict([
            # 创建一个名为'conv1_1'的卷积层，输入通道数为3，输出通道数为64，卷积核大小为3x3，步长为1，填充为1
            ('conv1_1', [3, 64, 3, 1, 1]),
            # 创建一个名为'conv1_2'的卷积层，输入通道数为64，输出通道数为64，卷积核大小为3x3，步长为1，填充为1
            ('conv1_2', [64, 64, 3, 1, 1]),
            # 创建一个名为'pool1_stage1'的最大池化层，核大小为2x2，步长为2，无填充
            ('pool1_stage1', [2, 2, 0]),
            # 创建一个名为'conv2_1'的卷积层，输入通道数为64，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv2_1', [64, 128, 3, 1, 1]),
            # 创建一个名为'conv2_2'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv2_2', [128, 128, 3, 1, 1]),
            # 创建一个名为'pool2_stage1'的最大池化层，核大小为2x2，步长为2，无填充
            ('pool2_stage1', [2, 2, 0]),
            # 创建一个名为'conv3_1'的卷积层，输入通道数为128，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
            ('conv3_1', [128, 256, 3, 1, 1]),
            # 创建一个名为'conv3_2'的卷积层，输入通道数为256，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
            ('conv3_2', [256, 256, 3, 1, 1]),
            # 创建一个名为'conv3_3'的卷积层，输入通道数为256，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
            ('conv3_3', [256, 256, 3, 1, 1]),
            # 创建一个名为'conv3_4'的卷积层，输入通道数为256，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
            ('conv3_4', [256, 256, 3, 1, 1]),
            # 创建一个名为'pool3_stage1'的最大池化层，核大小为2x2，步长为2，无填充
            ('pool3_stage1', [2, 2, 0]),
            # 创建一个名为'conv4_1'的卷积层，输入通道数为256，输出通道数为512，卷积核大小为3x3，步长为1，填充为1
            ('conv4_1', [256, 512, 3, 1, 1]),
            # 创建一个名为'conv4_2'的卷积层，输入通道数为512，输出通道数为512，卷积核大小为3x3，步长为1，填充为1
            ('conv4_2', [512, 512, 3, 1, 1]),
            # 创建一个名为'conv4_3_CPM'的卷积层，输入通道数为512，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
            ('conv4_3_CPM', [512, 256, 3, 1, 1]),
            # 创建一个名为'conv4_4_CPM'的卷积层，输入通道数为256，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv4_4_CPM', [256, 128, 3, 1, 1])
        ])
        # 定义第二个块的信息
        # 创建一个有序字典block1_1
        block1_1 = OrderedDict([
            # 创建一个名为'conv5_1_CPM_L1'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
            # 创建一个名为'conv5_2_CPM_L1'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
            # 创建一个名为'conv5_3_CPM_L1'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
            # 创建一个名为'conv5_4_CPM_L1'的卷积层，输入通道数为128，输出通道数为512，卷积核大小为1x1，步长为1，填充为0
            ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
            # 创建一个名为'conv5_5_CPM_L1'的卷积层，输入通道数为512，输出通道数为38，卷积核大小为1x1，步长为1，填充为0
            ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
        ])
        # 定义第三个块的信息
        # 创建一个有序字典block1_2
        block1_2 = OrderedDict([
            # 创建一个名为'conv5_1_CPM_L2'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
            # 创建一个名为'conv5_2_CPM_L2'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
            # 创建一个名为'conv5_3_CPM_L2'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
            # 创建一个名为'conv5_4_CPM_L2'的卷积层，输入通道数为128，输出通道数为512，卷积核大小为1x1，步长为1，填充为0
            ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
            # 创建一个名为'conv5_5_CPM_L2'的卷积层，输入通道数为512，输出通道数为19，卷积核大小为1x1，步长为1，填充为0
            ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
        ])
        # 将定义的块添加到blocks字典中
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2
        # 使用make_layers函数和block0的信息创建模型的第一部分
        self.model0 = make_layers(block0, no_relu_layers)
        # 对于2到6，创建相应的块并添加到blocks字典中
        # 对于2到6，我们将创建相应的块并添加到blocks字典中
        for i in range(2, 7):
            # 创建一个有序字典，其中包含了一系列卷积层和参数，这些层将用于构建模型的一部分
            blocks['block%d_1' % i] = OrderedDict([
                # 创建一个名为'Mconv1_stage%d_L1'的卷积层，输入通道数为185，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                # 创建一个名为'Mconv2_stage%d_L1'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv3_stage%d_L1'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv4_stage%d_L1'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv5_stage%d_L1'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv6_stage%d_L1'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为1x1，步长为1，填充为0
                ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                # 创建一个名为'Mconv7_stage%d_L1'的卷积层，输入通道数为128，输出通道数为38，卷积核大小为1x1，步长为1，填充为0
                ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
            ])
            # 创建另一个有序字典，其中包含了一系列卷积层和参数，这些层将用于构建模型的另一部分
            blocks['block%d_2' % i] = OrderedDict([
                # 创建一个名为'Mconv1_stage%d_L2'的卷积层，输入通道数为185，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                # 创建一个名为'Mconv2_stage%d_L2'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv3_stage%d_L2'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv4_stage%d_L2'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv5_stage%d_L2'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv6_stage%d_L2'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为1x1，步长为1，填充为0
                ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                # 创建一个名为'Mconv7_stage%d_L2'的卷积层，输入通道数为128，输出通道数为19，卷积核大小为1x1，步长为1，填充为0
                ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
            ])
        # 对于blocks字典中的每一个块，使用make_layers函数创建相应的模型部分
        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)
        # 将创建的模型部分赋值给类的属性
        # 将blocks字典中的'block1_1'赋值给self.model1_1，这是模型的一部分
        self.model1_1 = blocks['block1_1']
        # 将blocks字典中的'block2_1'赋值给self.model2_1，这是模型的一部分
        self.model2_1 = blocks['block2_1']
        # 将blocks字典中的'block3_1'赋值给self.model3_1，这是模型的一部分
        self.model3_1 = blocks['block3_1']
        # 将blocks字典中的'block4_1'赋值给self.model4_1，这是模型的一部分
        self.model4_1 = blocks['block4_1']
        # 将blocks字典中的'block5_1'赋值给self.model5_1，这是模型的一部分
        self.model5_1 = blocks['block5_1']
        # 将blocks字典中的'block6_1'赋值给self.model6_1，这是模型的一部分
        self.model6_1 = blocks['block6_1']
        # 将blocks字典中的'block1_2'赋值给self.model1_2，这是模型的一部分
        self.model1_2 = blocks['block1_2']
        # 将blocks字典中的'block2_2'赋值给self.model2_2，这是模型的一部分
        self.model2_2 = blocks['block2_2']
        # 将blocks字典中的'block3_2'赋值给self.model3_2，这是模型的一部分
        self.model3_2 = blocks['block3_2']
        # 将blocks字典中的'block4_2'赋值给self.model4_2，这是模型的一部分
        self.model4_2 = blocks['block4_2']
        # 将blocks字典中的'block5_2'赋值给self.model5_2，这是模型的一部分
        self.model5_2 = blocks['block5_2']
        # 将blocks字典中的'block6_2'赋值给self.model6_2，这是模型的一部分
        self.model6_2 = blocks['block6_2']

    # 定义前向传播函数，接受一个输入x
    def forward(self, x):
        # 将输入x通过第一部分的模型，得到输出out1
        out1 = self.model0(x)
        # 将out1通过第二部分的模型，得到输出out1_1
        out1_1 = self.model1_1(out1)
        # 将out1通过第三部分的模型，得到输出out1_2
        out1_2 = self.model1_2(out1)
        # 将out1_1、out1_2和out1在第一维度上拼接，得到输出out2
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        # 将out2通过第四部分的模型，得到输出out2_1
        out2_1 = self.model2_1(out2)
        # 将out2通过第五部分的模型，得到输出out2_2
        out2_2 = self.model2_2(out2)
        # 将out2_1、out2_2和out1在第一维度上拼接，得到输出out3
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        # 将out3通过第六部分的模型，得到输出out3_1
        out3_1 = self.model3_1(out3)
        # 将out3通过第七部分的模型，得到输出out3_2
        out3_2 = self.model3_2(out3)
        # 将out3_1、out3_2和out1在第一维度上拼接，得到输出out4
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        # 将out4通过第八部分的模型，得到输出out4_1
        out4_1 = self.model4_1(out4)
        # 将out4通过第九部分的模型，得到输出out4_2
        out4_2 = self.model4_2(out4)
        # 将out4_1、out4_2和out1在第一维度上拼接，得到输出out5
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        # 将out5通过第十部分的模型，得到输出out5_1
        out5_1 = self.model5_1(out5)
        # 将out5通过第十一部分的模型，得到输出out5_2
        out5_2 = self.model5_2(out5)
        # 将out5_1、out5_2和out1在第一维度上拼接，得到输出out6
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        # 将out6通过第十二部分的模型，得到输出out6_1
        out6_1 = self.model6_1(out6)
        # 将out6通过第十三部分的模型，得到输出out6_2
        out6_2 = self.model6_2(out6)
        # 返回out6_1和out6_2
        return out6_1, out6_2


# 定义一个名为handpose_model的类，该类继承自nn.Module，nn.Module是所有神经网络模块的基类
class handpose_model(nn.Module):
    # 定义初始化函数，接受self参数，self代表类的实例
    def __init__(self):
        # 调用父类的初始化函数
        super().__init__()
        # 定义一个列表，包含不需要ReLU激活函数的层的名称
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # 定义第一个块的信息，使用OrderedDict保证元素的插入顺序
        # 创建一个有序字典block1_0
        block1_0 = OrderedDict([
            # 创建一个名为'conv1_1'的卷积层，输入通道数为3，输出通道数为64，卷积核大小为3x3，步长为1，填充为1
            ('conv1_1', [3, 64, 3, 1, 1]),
            # 创建一个名为'conv1_2'的卷积层，输入通道数为64，输出通道数为64，卷积核大小为3x3，步长为1，填充为1
            ('conv1_2', [64, 64, 3, 1, 1]),
            # 创建一个名为'pool1_stage1'的最大池化层，核大小为2x2，步长为2，无填充
            ('pool1_stage1', [2, 2, 0]),
            # 创建一个名为'conv2_1'的卷积层，输入通道数为64，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv2_1', [64, 128, 3, 1, 1]),
            # 创建一个名为'conv2_2'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv2_2', [128, 128, 3, 1, 1]),
            # 创建一个名为'pool2_stage1'的最大池化层，核大小为2x2，步长为2，无填充
            ('pool2_stage1', [2, 2, 0]),
            # 创建一个名为'conv3_1'的卷积层，输入通道数为128，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
            ('conv3_1', [128, 256, 3, 1, 1]),
            # 创建一个名为'conv3_2'的卷积层，输入通道数为256，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
            ('conv3_2', [256, 256, 3, 1, 1]),
            # 创建一个名为'conv3_3'的卷积层，输入通道数为256，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
            ('conv3_3', [256, 256, 3, 1, 1]),
            # 创建一个名为'conv3_4'的卷积层，输入通道数为256，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
            ('conv3_4', [256, 256, 3, 1, 1]),
            # 创建一个名为'pool3_stage1'的最大池化层，核大小为2x2，步长为2，无填充
            ('pool3_stage1', [2, 2, 0]),
            # 创建一个名为'conv4_1'的卷积层，输入通道数为256，输出通道数为512，卷积核大小为3x3，步长为1，填充为1
            ('conv4_1', [256, 512, 3, 1, 1]),
            # 创建一个名为'conv4_2'的卷积层，输入通道数为512，输出通道数为512，卷积核大小为3x3，步长为1，填充为1
            ('conv4_2', [512, 512, 3, 1, 1]),
            # 创建一个名为'conv4_3'的卷积层，输入通道数为512，输出通道数为512，卷积核大小为3x3，步长为1，填充为1
            ('conv4_3', [512, 512, 3, 1, 1]),
            # 创建一个名为'conv4_4'的卷积层，输入通道数为512，输出通道数为512，卷积核大小为3x3，步长为1，填充为1
            ('conv4_4', [512, 512, 3, 1, 1]),
            # 创建一个名为'conv5_1'的卷积层，输入通道数为512，输出通道数为512，卷积核大小为3x3，步长为1，填充为1
            ('conv5_1', [512, 512, 3, 1, 1]),
            # 创建一个名为'conv5_2'的卷积层，输入通道数为512，输出通道数为512，卷积核大小为3x3，步长为1，填充为1
            ('conv5_2', [512, 512, 3, 1, 1]),
            # 创建一个名为'conv5_3_CPM'的卷积层，输入通道数为512，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            ('conv5_3_CPM', [512, 128, 3, 1, 1])
        ])

        # 定义第二个块的信息
        # 创建一个有序字典block1_1
        block1_1 = OrderedDict([
            # 创建一个名为'conv6_1_CPM'的卷积层，输入通道数为128，输出通道数为512，卷积核大小为1x1，步长为1，填充为0
            ('conv6_1_CPM', [128, 512, 1, 1, 0]),
            # 创建一个名为'conv6_2_CPM'的卷积层，输入通道数为512，输出通道数为22，卷积核大小为1x1，步长为1，填充为0
            ('conv6_2_CPM', [512, 22, 1, 1, 0])
        ])

        # 初始化一个空字典，用于存储各个块的信息
        blocks = {}
        # 将定义的块添加到blocks字典中
        # 将block1_0字典赋值给blocks字典的'block1_0'键，这是模型的一部分
        blocks['block1_0'] = block1_0
        # 将block1_1字典赋值给blocks字典的'block1_1'键，这是模型的一部分
        blocks['block1_1'] = block1_1

        # 对于2到6，创建相应的块并添加到blocks字典中
        # 对于2到6，我们将创建相应的块并添加到blocks字典中
        for i in range(2, 7):
            # 创建一个有序字典，其中包含了一系列卷积层和参数，这些层将用于构建模型的一部分
            blocks['block%d' % i] = OrderedDict([
                # 创建一个名为'Mconv1_stage%d'的卷积层，输入通道数为150，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv1_stage%d' % i, [150, 128, 7, 1, 3]),
                # 创建一个名为'Mconv2_stage%d'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv3_stage%d'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv4_stage%d'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv5_stage%d'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为7x7，步长为1，填充为3
                ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                # 创建一个名为'Mconv6_stage%d'的卷积层，输入通道数为128，输出通道数为128，卷积核大小为1x1，步长为1，填充为0
                ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                # 创建一个名为'Mconv7_stage%d'的卷积层，输入通道数为128，输出通道数为22，卷积核大小为1x1，步长为1，填充为0
                ('Mconv7_stage%d' % i, [128, 22, 1, 1, 0])
            ])

        # 对于blocks字典中的每一个块，使用make_layers函数创建相应的模型部分
        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        # 将创建的模型部分赋值给类的属性
        # 将blocks字典中的'block1_0'赋值给self.model1_0，这是模型的一部分
        self.model1_0 = blocks['block1_0']
        # 将blocks字典中的'block1_1'赋值给self.model1_1，这是模型的一部分
        self.model1_1 = blocks['block1_1']
        # 将blocks字典中的'block2'赋值给self.model2，这是模型的一部分
        self.model2 = blocks['block2']
        # 将blocks字典中的'block3'赋值给self.model3，这是模型的一部分
        self.model3 = blocks['block3']
        # 将blocks字典中的'block4'赋值给self.model4，这是模型的一部分
        self.model4 = blocks['block4']
        # 将blocks字典中的'block5'赋值给self.model5，这是模型的一部分
        self.model5 = blocks['block5']
        # 将blocks字典中的'block6'赋值给self.model6，这是模型的一部分
        self.model6 = blocks['block6']

    # 定义前向传播函数，接受一个输入x
    def forward(self, x):
        # 将输入x通过第一部分的模型，得到输出out1_0
        out1_0 = self.model1_0(x)
        # 将out1_0通过第二部分的模型，得到输出out1_1
        out1_1 = self.model1_1(out1_0)
        # 将out1_1和out1_0在第一维度上拼接，得到输出concat_stage2
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        # 将concat_stage2通过第三部分的模型，得到输出out_stage2
        out_stage2 = self.model2(concat_stage2)
        # 将out_stage2和out1_0在第一维度上拼接，得到输出concat_stage3
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        # 将concat_stage3通过第四部分的模型，得到输出out_stage3
        out_stage3 = self.model3(concat_stage3)
        # 将out_stage3和out1_0在第一维度上拼接，得到输出concat_stage4
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        # 将concat_stage4通过第五部分的模型，得到输出out_stage4
        out_stage4 = self.model4(concat_stage4)
        # 将out_stage4和out1_0在第一维度上拼接，得到输出concat_stage5
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        # 将concat_stage5通过第六部分的模型，得到输出out_stage5
        out_stage5 = self.model5(concat_stage5)
        # 将out_stage5和out1_0在第一维度上拼接，得到输出concat_stage6
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        # 将concat_stage6通过第七部分的模型，得到输出out_stage6
        out_stage6 = self.model6(concat_stage6)
        # 返回out_stage6
        return out_stage6
