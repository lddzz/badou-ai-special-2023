# 导入glob模块，用于查找符合特定规则的文件路径名
import glob
# 导入cv2模块，用于图像处理
import cv2
# 导入numpy模块，用于数组操作
import numpy as np
# 导入torch模块，用于深度学习
import torch
# 从model模块中导入UNet网络模型
from model.unet_model import UNet

# 主函数
if __name__ == "__main__":
    # 调用device方法判断是否有cuda设备，有则使用cuda，无则使用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化UNet网络模型，输入通道数为1，输出类别数为1
    net = UNet(n_channels=1, n_classes=1)
    # 将网络模型加载到设备上
    net.to(device=device)
    # 调用load_state_dict方法加载预训练模型参数
    net.load_state_dict(torch.load(f="best_model.pth", map_location=device))
    # 调用eval方法将网络模型设置为评估模式
    net.eval()
    # 调用glob.glob获取测试图片的路径
    tests_path = glob.glob("data/test/*.png")
    # 遍历每一张测试图片
    for test_path in tests_path:
        # 设置保存预测结果的路径
        save_res_path = test_path.split('.')[0] + "_res.png"
        # 读取测试图片
        img = cv2.imread(test_path)
        # 将测试图片转换为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 将二维的灰度图像重塑为四维的数组，以便可以作为网络模型的输入
        # 第一个维度是批量大小，这里是1，表示一次只输入一张图像
        # 第二个维度是通道数，这里是1，表示图像是灰度的，只有一个颜色通道
        # 第三个和第四个维度是图像的高度和宽度
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 将numpy数组转换为torch张量
        img_tensor = torch.from_numpy(img)
        # 将张量加载到设备上，并转换数据类型为float32
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 使用网络模型进行预测
        pred = net(img_tensor)
        # 将网络模型的预测结果从torch张量转换为numpy数组，并取出第一个元素
        pred = np.array(pred.data.cpu()[0])[0]
        # 将预测结果中大于等于0.5的值设置为255
        pred[pred >= 0.5] = 255
        # 将预测结果中小于0.5的值设置为0
        pred[pred < 0.5] = 0
        # 保存预测结果
        cv2.imwrite(save_res_path, pred)
