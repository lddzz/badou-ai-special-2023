import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
from skimage.measure import label

from src.model import handpose_model
from src import util


# 手部检测
class Hand(object):
    def __init__(self, model_path):
        self.model = handpose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        scale_search = [0.5, 1.0, 1.5, 2.0]
        boxsize = 368
        stride = 8
        padValue = 128
        thre = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        # 手部检测22个点
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 22))

        for i in range(len(multiplier)):
            scale = multiplier[i]
            image_resize = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            image_padding, padding = util.padRightDownCorner(image_resize, stride, padValue)
            image_input = np.transpose(np.float32(image_padding[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            image_input = np.ascontiguousarray(image_input)

            data = torch.from_numpy(image_input).float()
            if torch.cuda.is_available():
                data = data.cuda()
            with torch.no_grad():
                output = self.model(data).cpu().numpy()

            # heatmaps(关节点特征图)：提取特征的结果进行整形，去padding，恢复成和输入一样的格式
            heatmap = np.transpose(np.squeeze(output), (1, 2, 0))
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:image_padding.shape[0] - padding[2], :image_padding.shape[1] - padding[3], :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap / len(multiplier)

        all_peaks = []
        # 对手部的每个关节点进行判断，以阈值为依据
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
            # 全部小于阈值(小于阈值的都是0)
            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                continue
            label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
            max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = util.npmax(map_ori)
            all_peaks.append([x, y])
        return np.array(all_peaks)

# if __name__ == "__main__":
#     hand_estimation = Hand('../model/hand_pose_model.pth')
#
#     # test_image = '../images/hand.jpg'
#     test_image = '../images/person.jpg'
#     oriImg = cv2.imread(test_image)  # B,G,R order
#     peaks = hand_estimation(oriImg)
#     canvas = util.draw_handpose(oriImg, peaks, True)
#     cv2.imshow('', canvas)
#     cv2.waitKey(0)