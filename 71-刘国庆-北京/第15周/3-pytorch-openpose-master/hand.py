# 导入cv2库，这是一个开源的计算机视觉库，提供了图像处理和计算机视觉的各种算法
import cv2
# 导入numpy库，这是一个用于处理数组，矩阵等数据结构的库，提供了大量的数学函数
import numpy as np
# 导入torch库，torch是一个开源的机器学习库，提供了广泛的模块和类，如张量操作，自动求导等
import torch
# 从scipy.ndimage.filters导入gaussian_filter函数，这个函数用于对图像进行高斯滤波
from scipy.ndimage.filters import gaussian_filter
# 从skimage.measure导入label函数，这个函数用于标记二值图像中的连通区域
from skimage.measure import label
# 从src目录下的util模块导入所有函数和类，这个模块可能包含了一些实用的函数和类
from src import util
# 从src.model模块导入handpose_model类，这个类可能是用于手部姿态估计的模型
from src.model import handpose_model


# 定义一个名为Hand的类
class Hand(object):
    # 类的初始化函数，接收一个参数model_path，这个参数是模型的路径
    def __init__(self,model_path):
        # 创建handpose_model的实例
        self.model=handpose_model()
        # 检查是否有可用的CUDA设备，如果有，则将模型放到CUDA设备上
        if torch.cuda.is_available():
            self.model=self.model.cuda()
        # 加载模型参数model_dict
        model_dict=util.transfer(self.model,torch.load(model_path))
        # 将加载的参数赋值给模型
        self.model.load_state_dict(model_dict)
        # 将模型设置为评估模式，这意味着在这个模式下，模型的行为会有所不同，比如dropout和batchnorm
        self.model.eval()


    # 定义__call__方法，接收一个参数image，这个参数是原始图像
    # 这个方法的作用是对原始图像进行处理，然后返回峰值,即手部关键点
    def __call__(self,image):
        # Step 1: 定义一些参数
        # 尺度列表scale_search
        scale_search=[0.5,1.0,1.5,2.0]
        # 图像的大小boxsize
        boxsize=368
        # 步长stride
        stride=8
        # 填充值padValue
        padValue=128
        # 阈值 thre
        thre=0.05
        # Step 2: 计算每个尺度对应的乘数multiplier
        # 计算每个尺度对应的乘数:
        # 乘数 = 尺度 * 图像大小 /图像的高度
        # image.shape[0]表示图像的高度
        multiplier=[x*boxsize/image.shape[0] for x in scale_search]
        # Step 3: 初始化一个全零的三维数组
        # 初始化一个全零的三维数组，用于存储热图的平均值
        # np.zeros函数用于创建一个指定形状和数据类型的全零数组
        # image.shape[0]: 数组的第一个维度: 图像的高度
        # image.shape[1]: 数组的第二个维度:
        # 22: 数组的第三个维度: 手部关键点的数量
        heatmap_avg=np.zeros((image.shape[0],image.shape[1],22))
        # Step 4: 遍历每个乘数
        # 遍历每个乘数
        for m in range(len(multiplier)):
            # 获取当前的乘数
            scale=multiplier[m]
            # 将原始图像按照当前的乘数进行缩放
            # cv2.resize函数用于对图像进行缩放
            # image：原始图像
            # (0, 0)：输出图像的大小，如果为(0, 0)，表示输出图像的大小由fx和fy决定
            # fx=scale：沿水平轴的缩放系数
            # fy=scale：沿垂直轴的缩放系数
            # interpolation=cv2.INTER_CUBIC：插值方法，这里使用双三次插值
            imageToTest=cv2.resize(image,(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            # 对缩放后的图像进行填充
            # util.padRightDownCorner函数用于对图像进行填充
            # imageToTest：缩放后的图像
            # stride：步长
            # padValue：填充值
            imageToTest_padded,pad=util.padRightDownCorner(imageToTest,stride,padValue)
            # 对填充后的图像进行处理，首先将其转换为浮点类型，然后增加一个维度，然后进行转置，最后进行归一化和中心化
            # imageToTest_padded[:, :, :, np.newaxis]: 填充后的图像增加一个维度
            # np.float32：转换为浮点类型
            # (3, 2, 0, 1): 转置,将通道维度放在最前面,然后是高度和宽度,最后是批大小
            # np.transpose：转置
            # / 256 - 0.5：归一化和中心化
            # im：处理后的图像
            # np.transpose：转置
            im=np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]),(3,2,0,1))/256-0.5
            # 将处理后的图像转换为连续的数组
            im=np.ascontiguousarray(im)
            # 将处理后的图像转换为torch的张量
            data=torch.from_numpy(im).float()
            # 如果有可用的CUDA设备，就将数据放到CUDA设备上
            if torch.cuda.is_available():
                data=data.cuda()
            # 使用torch.no_grad()上下文管理器，表示接下来的计算不需要计算梯度，不会进行反向传播
            with torch.no_grad():
                # 将数据输入模型，得到输出，然后将输出转换为numpy数组
                output=self.model(data).cpu().numpy()
            # Step 5: 提取输出，然后进行缩放，然后去掉填充
            # 提取输出，然后进行缩放，然后去掉填充
            # np.squeeze(output): 去掉维度为1的维度
            # (1, 2, 0): 转置,将通道维度放在最后面,然后是高度和宽度
            # np.transpose：转置
            heatmap=np.transpose(np.squeeze(output),(1,2,0))
            # 对热图进行缩放: 将热图的大小缩放为原始图像的大小,然后去掉填充,最后进行插值,得到最终的热图
            # cv2.resize函数用于对图像进行缩放
            # (0, 0): 输出图像的大小,如果为(0, 0),表示输出图像的大小由fx和fy决定
            # fx=stride: 沿水平轴的缩放系数
            # fy=stride: 沿垂直轴的缩放系数
            # interpolation=cv2.INTER_CUBIC: 插值方法,这里使用双三次插值
            heatmap=cv2.resize(heatmap,(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            # :imageToTest_padded.shape[0] - pad[2]: 去掉填充
            # :imageToTest_padded.shape[1] - pad[3]: 去掉填充
            heatmap=heatmap[:imageToTest_padded.shape[0]-pad[2],:imageToTest_padded.shape[1]-pad[3],:]
            # 对热图进行插值,得到最终的热图,然后将其转换为numpy数组,最后将其添加到热图的平均值上
            # cv2.resize函数用于对图像进行缩放
            # image.shape[1]: 输出图像的宽度
            # image.shape[0]: 输出图像的高度
            # interpolation=cv2.INTER_CUBIC: 插值方法,这里使用双三次插值
            heatmap=cv2.resize(heatmap,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_CUBIC)
            # 将当前的热图加到热图的平均值上
            heatmap_avg+=heatmap/len(multiplier)

        # Step 6: 初始化一个空列表
        # 初始化一个空列表，用于存储所有的峰值: 手部关键点
        all_peaks=[]
        # Step 7: 遍历每个部分
        # 遍历每个部分:21: 手部关键点的数量
        for part in range(22):
            # 获取当前部分的热图
            # heatmap_avg[:, :, part]: 热图的第part个通道
            map_roi=heatmap_avg[:,:,part]
            # 对当前部分的热图进行高斯滤波: 这里使用的是sigma=3
            one_heatmap=gaussian_filter(map_roi,sigma=3)
            # 将滤波后的热图转换为二值图像: 如果大于阈值,就设置为1;否则,设置为0
            binary=np.ascontiguousarray(one_heatmap>thre,dtype=np.uint8)
            # Step 8: 对二值图像进行标记
            # 如果二值图像中所有的值都小于阈值，就将[0, 0]添加到峰值列表中，然后继续下一次循环
            if np.sum(binary)==0:
                all_peaks.append([0,0])
                continue
            # 对二值图像进行标记，得到标记图像label_img和标记的数量label_numbers
            # label: 标记二值图像中的连通区域
            # binary: 二值图像
            # return_num=True: 返回标记的数量
            # connectivity=binary.ndim: 连通性
            label_img,label_numbers=label(binary,return_num=True,connectivity=binary.ndim)
            # Step 9: 找到标记图像中的最大索引
            # 找到标记图像中的最大索引
            # range(1, label_numbers + 1)]) + 1: 遍历1到label_numbers + 1,然后找到最大的索引
            # np.sum(map_ori[label_img == i]): 找到标记图像中索引为i的部分，然后计算其热图的和
            # np.argmax: 找到最大值的索引
            max_index=np.argmax([np.sum(map_roi[label_img==i]) for i in range(1,label_numbers+1)])+1
            # 将标记图像中不等于最大索引的部分设置为0
            label_img[label_img!=max_index]=0
            # 将原始热图中对应标记图像为0的部分设置为0
            map_roi[label_img==0]=0
            # Step 10: 找到处理后的热图中的最大值的位置
            # 找到处理后的热图中的最大值的位置
            y, x = util.npmax(map_roi)
            # 将最大值的位置添加到手部关键点列表中
            all_peaks.append([x, y])
        # Step 11: 将手部关键点列表转换为numpy数组，然后返回
        # 将手部关键点列表转换为numpy数组，然后返回
        return np.array(all_peaks)
