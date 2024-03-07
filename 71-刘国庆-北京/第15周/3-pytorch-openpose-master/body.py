# 导入Python的math库，提供数学函数和常量
import math
# 导入cv2库，这是一个开源的计算机视觉库，提供了图像处理和计算机视觉的各种算法
import cv2
# 导入matplotlib的pyplot模块，这是一个用于创建图形的库
import matplotlib.pyplot as plt
# 导入numpy库，这是一个用于处理数组，矩阵等数据结构的库，提供了大量的数学函数
import numpy as np
# 导入torch库，torch是一个开源的机器学习库，提供了广泛的模块和类，如张量操作，自动求导等
import torch
# 从scipy.ndimage.filters导入gaussian_filter函数，这个函数用于对图像进行高斯滤波
from scipy.ndimage.filters import gaussian_filter
# 从src目录下的util模块导入所有函数和类，这个模块可能包含了一些实用的函数和类
from src import util
# 从src.model模块导入bodypose_model类，这个类可能是用于人体姿态估计的模型
from src.model import bodypose_model


# 定义一个名为Body的类
class Body(object):
    # 类的初始化函数，接收一个参数model_path，这个参数是模型的路径
    def __init__(self, model_path):
        # 创建一个bodypose_model的实例，并赋值给self.model
        self.model = bodypose_model()
        # 检查是否有可用的CUDA设备，如果有，则将模型转移到CUDA设备上
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # 使用torch.load函数加载模型路径下的模型，并使用util.transfer函数将模型转移到当前设备上，然后将结果赋值给model_dict
        model_dict = util.transfer(self.model, torch.load(model_path))
        # 使用模型的load_state_dict方法，将model_dict中的状态加载到模型中
        self.model.load_state_dict(model_dict)
        # 将模型设置为评估模式，这意味着在此模式下，模型的某些特定层（如Dropout，BatchNorm等）会以不同的方式运行
        self.model.eval()

    # Step 1: 定义__call__方法，接收一个参数oriImg，这个参数是原始图像
    def __call__(self, oriImg):
        # Step 2: 定义搜索比例，这个比例用于调整图像的大小
        scale_search = [0.5]
        # Step 3: 定义boxsize，这个值用于确定图像的大小
        boxsize = 368
        # Step 4: 定义stride，这个值用于确定图像的步长
        stride = 8
        # Step 5: 定义padValue，这个值用于确定图像的填充值
        padValue = 128
        # Step 6: 定义thre1和thre2，这两个值用于确定热图和PAF的阈值
        thre1 = 0.1
        thre2 = 0.05
        # Step 7: 计算multiplier乘数，这个值用于调整图像的大小
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        # Step 8: 初始化heatmap_avg和paf_avg，这两个值用于存储热图和PAF的平均值
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        # Step 9: 遍历每个乘数，对每个scale进行处理
        for m in range(len(multiplier)):
            # Step 10: 获取当前的scale
            scale = multiplier[m]
            # Step 11: 调整图像的大小
            # 将原始图像按照当前的乘数进行缩放
            # cv2.resize函数用于对图像进行缩放
            # image：原始图像
            # (0, 0)：输出图像的大小，如果为(0, 0)，表示输出图像的大小由fx和fy决定
            # fx=scale：沿水平轴的缩放系数
            # fy=scale：沿垂直轴的缩放系数
            # interpolation=cv2.INTER_CUBIC：插值方法，这里使用双三次插值
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # Step 12: 对图像进行填充
            # 对缩放后的图像进行填充
            # util.padRightDownCorner函数用于对图像进行填充
            # imageToTest：缩放后的图像
            # stride：步长
            # padValue：填充值
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            # Step 13: 对图像进行预处理，包括转置和归一化
            # imageToTest_padded[:, :, :, np.newaxis]: 填充后的图像增加一个维度
            # np.float32：转换为浮点类型
            # (3, 2, 0, 1): 转置,将通道维度放在最前面,然后是高度和宽度,最后是批大小
            # np.transpose：转置
            # / 256 - 0.5：归一化和中心化
            # im：处理后的图像
            # np.transpose：转置
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            # Step 14: 将图像转换为连续的数组
            im = np.ascontiguousarray(im)
            # Step 15: 将图像转换为torch张量
            data = torch.from_numpy(im).float()
            # Step 16: 如果有可用的CUDA设备，将数据转移到CUDA设备上
            if torch.cuda.is_available():
                data = data.cuda()
            # Step 17: 使用模型进行预测，获取Mconv7_stage6_L1和Mconv7_stage6_L2
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
            # Step 18: 将Mconv7_stage6_L1和Mconv7_stage6_L2转换为numpy数组
            Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
            Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()
            # Step 19: 获取heatmap，这是模型的输出之一，表示热图
            heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))  # output 1 is heatmaps
            # Step 20: 调整heatmap的大小，并去除填充
            # cv2.resize函数用于对图像进行缩放
            # (0, 0): 输出图像的大小,如果为(0, 0),表示输出图像的大小由fx和fy决定
            # fx=stride: 沿水平轴的缩放系数
            # fy=stride: 沿垂直轴的缩放系数
            # interpolation=cv2.INTER_CUBIC: 插值方法,这里使用双三次插值
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            # :imageToTest_padded.shape[0] - pad[2]: 去掉填充
            # :imageToTest_padded.shape[1] - pad[3]: 去掉填充
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            # cv2.resize函数用于对图像进行缩放
            # image.shape[1]: 输出图像的宽度
            # image.shape[0]: 输出图像的高度
            # interpolation=cv2.INTER_CUBIC: 插值方法,这里使用双三次插值
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            # Step 21: 获取paf，这是模型的另一个输出，表示PAF
            # 提取输出，然后进行缩放，然后去掉填充
            # np.squeeze(output): 去掉维度为1的维度
            # (1, 2, 0): 转置,将通道维度放在最后面,然后是高度和宽度
            # np.transpose：转置
            paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))
            # Step 22: 调整paf的大小，并去除填充
            # cv2.resize函数用于对图像进行缩放
            # (0, 0): 输出图像的大小,如果为(0, 0),表示输出图像的大小由fx和fy决定
            # fx=stride: 沿水平轴的缩放系数
            # fy=stride: 沿垂直轴的缩放系数
            # interpolation=cv2.INTER_CUBIC: 插值方法,这里使用双三次插值
            paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            # :imageToTest_padded.shape[0] - pad[2]: 去掉填充
            # :imageToTest_padded.shape[1] - pad[3]: 去掉填充
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            # cv2.resize函数用于对图像进行缩放
            # image.shape[1]: 输出图像的宽度
            # image.shape[0]: 输出图像的高度
            # interpolation=cv2.INTER_CUBIC: 插值方法,这里使用双三次插值
            paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            # Step 23: 更新heatmap_avg和paf_avg
            heatmap_avg += heatmap_avg + heatmap / len(multiplier)
            paf_avg += + paf / len(multiplier)
        # Step 24: 初始化all_peaks和peak_counter，用于存储所有的峰值和峰值计数器
        all_peaks = []
        peak_counter = 0
        # Step 25: 遍历heatmap_avg的每一个部分，找出所有的峰值
        # 遍历heatmap_avg的每一个部分，找出所有的峰值
        for part in range(18):
            # 获取heatmap_avg中的第part个元素，赋值给map_ori
            map_ori = heatmap_avg[:, :, part]
            # 对map_ori进行高斯滤波，赋值给one_heatmap
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            # 创建一个与one_heatmap形状相同的全0数组，赋值给map_left
            map_left = np.zeros(one_heatmap.shape)
            # 将one_heatmap的第1行到最后一行赋值给map_left的第2行到最后一行
            map_left[1:, :] = one_heatmap[:-1, :]
            # 创建一个与one_heatmap形状相同的全0数组，赋值给map_right
            map_right = np.zeros(one_heatmap.shape)
            # 将one_heatmap的第0行到倒数第二行赋值给map_right的第1行到最后一行
            map_right[:-1, :] = one_heatmap[1:, :]
            # 创建一个与one_heatmap形状相同的全0数组，赋值给map_up
            map_up = np.zeros(one_heatmap.shape)
            # 将one_heatmap的第1列到最后一列赋值给map_up的第2列到最后一列
            map_up[:, 1:] = one_heatmap[:, :-1]
            # 创建一个与one_heatmap形状相同的全0数组，赋值给map_down
            map_down = np.zeros(one_heatmap.shape)
            # 将one_heatmap的第0列到倒数第二列赋值给map_down的第1列到最后一列
            map_down[:, :-1] = one_heatmap[:, 1:]
            # 计算one_heatmap是否大于等于map_left、map_right、map_up、map_down和thre1，结果赋值给peaks_binary
            peaks_binary = np.logical_and.reduce(
                (
                    one_heatmap >= map_left,
                    one_heatmap >= map_right,
                    one_heatmap >= map_up,
                    one_heatmap >= map_down,
                    one_heatmap > thre1
                )
            )
            # 获取peaks_binary中非零元素的坐标，坐标的第一维和第二维组成一个元组，所有的元组组成一个列表，赋值给peaks
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
            # 遍历peaks，对每个元素，将元素和map_ori中对应位置的元素组成一个新的元组，所有的新元组组成一个列表，赋值给peaks_with_score
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            # 创建一个从peak_counter到peak_counter + len(peaks)的范围，赋值给peak_id
            peak_id = range(peak_counter, peak_counter + len(peaks))
            # 遍历peak_id，对每个元素，将peaks_with_score中对应位置的元素和元素本身组成一个新的元组，所有的新元组组成一个列表，赋值给peaks_with_score_and_id
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]
            # 将peaks_with_score_and_id添加到all_peaks的末尾
            all_peaks.append(peaks_with_score_and_id)
            # 更新peak_counter，加上peaks的长度
            peak_counter += len(peaks)
        # Step 26: 定义limbSeq和mapIdx，这两个值用于确定身体的部分和映射的索引
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
                  [55, 56], [37, 38], [45, 46]]
        # Step 27: 初始化connection_all和special_k，用于存储所有的连接和特殊的k
        connection_all = []
        special_k = []
        mid_num = 10
        # Step 28: 遍历mapIdx，对每个k进行处理
        # 遍历mapIdx的长度，mapIdx是一个列表，包含了身体部位之间的连接关系
        for k in range(len(mapIdx)):
            # 获取paf_avg中的第k个元素的第19个以后的所有元素，赋值给score_mid，paf_avg是一个数组，包含了所有的PAF
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            # 获取all_peaks中的第limbSeq[k][0] - 1个元素，赋值给candA，all_peaks是一个列表，包含了所有的峰值
            candA = all_peaks[limbSeq[k][0] - 1]
            # 获取all_peaks中的第limbSeq[k][1] - 1个元素，赋值给candB
            candB = all_peaks[limbSeq[k][1] - 1]
            # 计算candA的长度，赋值给nA
            nA = len(candA)
            # 计算candB的长度，赋值给nB
            nB = len(candB)
            # 获取limbSeq中的第k个元素，赋值给indexA和indexB，limbSeq是一个列表，包含了身体部位的连接顺序
            indexA, indexB = limbSeq[k]
            # 检查nA和nB是否都不等于0
            if nA != 0 and nB != 0:
                # 初始化connection_candidate为空列表，用于存储所有的连接候选
                connection_candidate = []
                # 遍历candA的长度
                for i in range(nA):
                    # 遍历candB的长度
                    for j in range(nB):
                        # 计算candB中的第j个元素的前两个元素和candA中的第i个元素的前两个元素的差，赋值给vec
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        # 计算vec的模，赋值给norm
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        # 将norm的最小值设置为0.001
                        norm = max(0.001, norm)
                        # 将vec除以norm，赋值给vec
                        vec = np.divide(vec, norm)
                        # 创建一个列表，包含了从candA中的第i个元素的第0个元素到candB中的第j个元素的第0个元素的等差数列，和从candA中的第i个元素的第1个元素到candB中的第j个元素的第1个元素的等差数列，赋值给startend
                        startend = list(zip(
                            np.linspace(candA[i][0], candB[j][0], num=mid_num),
                            np.linspace(candA[i][1], candB[j][1], num=mid_num)
                        ))
                        # 计算score_mid中的第startend[I][1]行、第startend[I][0]列、第0个元素，赋值给vec_x
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in
                                          range(len(startend))])
                        # 计算score_mid中的第startend[I][1]行、第startend[I][0]列、第1个元素，赋值给vec_y
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in
                                          range(len(startend))])
                        # 计算vec_x和vec[0]的乘积，加上vec_y和vec[1]的乘积，赋值给score_midpts
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        # 计算score_midpts的平均值，加上0.5 * oriImg.shape[0] / norm - 1的最小值，赋值给score_with_dist_prior
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        # 计算score_midpts大于thre2的元素的数量是否大于score_midpts的长度的80%
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        # 检查score_with_dist_prior是否大于0
                        criterion2 = score_with_dist_prior > 0
                        # 如果criterion1和criterion2都满足
                        if criterion1 and criterion2:
                            # 将[i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]]添加到connection_candidate中
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
                # 将connection_candidate按照第2个元素的值进行降序排序
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                # 初始化connection为0的5列的数组
                connection = np.zeros((0, 5))
                # 遍历connection_candidate的长度
                for c in range(len(connection_candidate)):
                    # 获取connection_candidate中的第c个元素的前三个元素，赋值给i、j和s
                    i, j, s = connection_candidate[c][0:3]
                    # 检查i是否不在connection的第3列中，和j是否不在connection的第4列中
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        # 将[candA[i][3], candB[j][3], s, i, j]添加到connection的末尾
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        # 检查connection的长度是否大于等于nA和nB的最小值
                        if len(connection) >= min(nA, nB):
                            # 如果满足条件，跳出循环
                            break
                # 将connection添加到connection_all的末尾，connection_all是一个列表，用于存储所有的连接
                connection_all.append(connection)
            # 如果nA等于0或者nB等于0
            else:
                # 将k添加到special_k的末尾，special_k是一个列表，用于存储特殊的k值
                special_k.append(k)
                # 将空列表添加到connection_all的末尾
                connection_all.append([])
        # Step 29: 初始化subset，用于存储所有的子集
        subset = -1 * np.ones((0, 20))
        # Step 30: 将all_peaks转换为numpy数组
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        # Step 31: 遍历mapIdx，对每个k进行处理
        # 遍历mapIdx的长度，mapIdx是一个列表，包含了身体部位之间的连接关系
        for k in range(len(mapIdx)):
            # 检查当前的k是否在special_k中，special_k是一个列表，包含了特殊的k值
            if k not in special_k:
                # 获取connection_all中的第k个元素的第一列，赋值给partAs，connection_all是一个列表，包含了所有的连接
                partAs = connection_all[k][:, 0]
                # 获取connection_all中的第k个元素的第二列，赋值给partBs
                partBs = connection_all[k][:, 1]
                # 获取limbSeq中的第k个元素，减1后赋值给indexA和indexB，limbSeq是一个列表，包含了身体部位的连接顺序
                indexA, indexB = np.array(limbSeq[k]) - 1
                # 遍历connection_all中的第k个元素的长度
                for i in range(len(connection_all[k])):
                    # 初始化found为0，用于记录找到的数量
                    found = 0
                    # 初始化subset_idx为[-1, -1]，用于记录找到的索引
                    subset_idx = [-1, -1]
                    # 遍历subset的长度，subset是一个数组，用于存储所有的子集
                    for j in range(len(subset)):
                        # 检查subset中的第j个元素的indexA是否等于partAs中的第i个元素，或者subset中的第j个元素的indexB是否等于partBs中的第i个元素
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            # 如果满足条件，将j赋值给subset_idx中的第found个元素，并将found加1
                            subset_idx[found] = j
                            found += 1
                    # 如果found等于1
                    if found == 1:
                        # 获取subset_idx中的第0个元素，赋值给j
                        j = subset_idx[0]
                        # 检查subset中的第j个元素的indexB是否不等于partBs中的第i个元素
                        if subset[j][indexB] != partBs[i]:
                            # 如果满足条件，将partBs中的第i个元素赋值给subset中的第j个元素的indexB
                            subset[j][indexB] = partBs[i]
                            # 将subset中的第j个元素的倒数第一个元素加1
                            subset[j][-1] += 1
                            # 将subset中的第j个元素的倒数第二个元素加上candidate中的第partBs[i]个元素的第2个元素和connection_all中的第k个元素的第i个元素的第2个元素
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    # 如果found等于2
                    elif found == 2:
                        # 获取subset_idx中的第0个和第1个元素，赋值给j1和j2
                        j1, j2 = subset_idx
                        # 计算subset中的第j1个元素和第j2个元素是否大于等于0，然后转换为整数，最后去掉最后两个元素，赋值给membership
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        # 检查membership中等于2的元素的数量是否等于0
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            # 如果满足条件，将subset中的第j2个元素加1后加到subset中的第j1个元素上，然后将subset中的第j2个元素的最后两个元素加到subset中的第j1个元素的最后两个元素上
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            # 将connection_all中的第k个元素的第i个元素的第2个元素加到subset中的第j1个元素的倒数第二个元素上
                            subset[j1][-2] += connection_all[k][i][2]
                            # 从subset中删除第j2个元素
                            subset = np.delete(subset, j2, 0)
                        else:
                            # 如果不满足条件，将partBs中的第i个元素赋值给subset中的第j1个元素的indexB
                            subset[j1][indexB] = partBs[i]
                            # 将subset中的第j1个元素的倒数第一个元素加1
                            subset[j1][-1] += 1
                            # 将subset中的第j1个元素的倒数第二个元素加上candidate中的第partBs[i]个元素的第2个元素和connection_all中的第k个元素的第i个元素的第2个元素
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    # 如果found等于0并且k小于17
                    elif not found and k < 17:
                        # 创建一个长度为20的全-1数组，赋值给row
                        row = -1 * np.ones(20)
                        # 将partAs中的第i个元素赋值给row的indexA
                        row[indexA] = partAs[i]
                        # 将partBs中的第i个元素赋值给row的indexB
                        row[indexB] = partBs[i]
                        # 将2赋值给row的倒数第一个元素
                        row[-1] = 2
                        # 将candidate中的第connection_all[k][i, :2]个元素的第2个元素和connection_all中的第k个元素的第i个元素的第2个元素的和赋值给row的倒数第二个元素
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        # 将row添加到subset的末尾
                        subset = np.vstack([subset, row])
        # 初始化deleteIdx为空列表，用于存储需要删除的索引
        deleteIdx = []
        # 遍历subset的长度
        for i in range(len(subset)):
            # 检查subset中的第i个元素的倒数第一个元素是否小于4，或者subset中的第i个元素的倒数第二个元素除以subset中的第i个元素的倒数第一个元素是否小于0.4
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                # 如果满足条件，将i添加到deleteIdx中
                deleteIdx.append(i)
        # 从subset中删除deleteIdx中的所有元素
        subset = np.delete(subset, deleteIdx, axis=0)
        # 返回candidate和subset
        return candidate, subset


# 检查当前是否在主程序中运行，而不是作为模块导入
if __name__ == "__main__":
    # 创建Body类的实例，传入模型路径，用于人体姿态估计
    body_estimation = Body('../model/body_pose_model.pth')
    # 定义测试图像的路径
    test_image = '../images/ski.jpg'
    # 使用cv2库的imread函数读取测试图像
    oriImg = cv2.imread(test_image)
    # 调用body_estimation实例的__call__方法，传入原始图像，返回候选关键点和子集
    candidate, subset = body_estimation(oriImg)
    # 调用util模块的draw_bodypose函数，传入原始图像、候选关键点和子集，返回绘制了人体姿态的图像
    canvas = util.draw_bodypose(oriImg, candidate, subset)
    # 使用matplotlib的pyplot模块的imshow函数显示图像，注意这里将图像的颜色通道从BGR转换为RGB
    plt.imshow(canvas[:, :, [2, 1, 0]])
    # 使用matplotlib的pyplot模块的show函数显示图像
    plt.show()
