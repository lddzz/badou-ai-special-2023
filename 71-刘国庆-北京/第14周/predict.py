# 这个`predict.py`文件的主要作用是使用预训练的Mask R-CNN模型进行图像分割或物体检测。具体步骤如下：
# 1. 导入所需的模块和类，包括PIL库的Image模块和自定义的MASK_RCNN类
# 2. 创建MASK_RCNN类的一个实例对象
# 3. 打开一个图像文件，这里是'img/street.jpg'
# 4. 使用MASK_RCNN对象的`detect_image`方法对打开的图像进行检测或分割
# 5. 最后，调用MASK_RCNN对象的`close_session`方法，可能用于清理或关闭与模型相关的资源
# 总的来说，这个文件是一个简单的脚本，用于展示如何使用MASK_RCNN类进行图像分割或物体检测

# 导入图像处理库 PIL 中的 Image 模块
from PIL import Image
# 从自定义的 mask_rcnn 模块中导入 MASK_RCNN 类或对象
from mask_rcnn import MASK_RCNN

# 创建 MASK_RCNN 类的一个实例对象，通常用于图像分割或物体检测
mask_rcnn = MASK_RCNN()
# 打开名为 'street.jpg' 的图像文件，该文件位于 'img' 文件夹中
image = Image.open("img/street.jpg")
# 调用 MASK_RCNN 对象的 detect_image 方法，对打开的图像进行检测或分割
mask_rcnn.detect_image(image)
# 调用 MASK_RCNN 对象的 close_session 方法，可能用于清理或关闭与模型相关的资源
mask_rcnn.close_session()
