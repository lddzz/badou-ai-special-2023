# 导入所需模块
from PIL import Image
from frcnn import FRCNN

# 创建FRCNN模型实例
frcnn = FRCNN()

# 尝试打开图像文件
image = Image.open("img/street.jpg")
# 如果图像成功打开，调用FRCNN模型进行目标检测
r_image = frcnn.detect_image(image)
# 显示检测结果的图像
r_image.show()

# 关闭FRCNN模型的会话，释放资源
frcnn.close_session()
