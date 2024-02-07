# 导入操作系统模块
import os
# 导入处理数值计算的 NumPy 库
import numpy as np
# 导入 TensorFlow 深度学习框架
import tensorflow as tf
# 从 Python Imaging Library (PIL) 模块中导入图像处理相关的类和函数
from PIL import Image, ImageFont, ImageDraw
# 导入配置模块
import config
# 从 utils 模块中导入辅助函数
from utils import letterbox_image, load_weights
# 从 yolo_predict 模块中导入 yolo_predictor 类
from yolo_predict import yolo_predictor

# 设置使用的 GPU 索引
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index


# model_path: 模型路径，当使用yolo_weights无用
#         image_path: 图片路径
def detect(image_path, model_path, yolo_weights=None):
    # ---------------------------------------#
    #   Step 1:图片预处理
    # ---------------------------------------#
    # 打开指定路径的图像文件，创建一个 PIL 图像对象
    image = Image.open(image_path)
    # 使用 letterbox_image 函数将图像调整为指定大小 (416, 416)，保持原始图像的纵横比
    resize_image = letterbox_image(image, (416, 416))
    # 将调整后的图像转换为 NumPy 数组，并指定数据类型为 float32
    image_data = np.array(resize_image, dtype=np.float32)
    # 将图像数据的值归一化到 [0, 1] 的范围，通过除以 255
    image_data /= 255.
    # 在数组的第一个维度上添加一个维度，将其变成形状为 (1, height, width, channels) 的四维数组
    # 这是因为模型一般期望输入是一个批次的图像
    image_data = np.expand_dims(image_data, axis=0)

    # ---------------------------------------#
    #   Step 2:图片输入
    # ---------------------------------------#
    # 创建一个 TensorFlow 占位符input_image_shape，用于传递原图的大小信息，数据类型为 int32，形状为 (2,)
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2,))
    # 创建一个 TensorFlow 占位符input_image，用于传递图像数据，形状为 [批次大小, 高度, 宽度, 通道数]，数据类型为 float32
    # 这里的高度和宽度为固定值 416，通道数为 3（RGB图像）
    input_image = tf.placeholder(dtype=tf.float32, shape=[None, 416, 416, 3])

    # ---------------------------------------#
    #   Step 3:YOLO 预测器初始化
    # ---------------------------------------#
    # 进入yolo_predictor进行预测，yolo_predictor是用于预测的一个对象
    # 创建一个 YOLO 预测器对象，用于进行目标检测
    # 通过传递一些配置参数，如
    # 目标置信度阈值config.obj_threshold
    # 非极大值抑制阈值config.nms_threshold
    # 类别信息文件路径config.classes_path
    # 锚框信息文件路径config.anchors_path
    predictor = yolo_predictor(config.obj_threshold,
                               config.nms_threshold,
                               config.classes_path,
                               config.anchors_path)

    # ---------------------------------------#
    #   Step 4: 图片预测
    # ---------------------------------------#
    # 创建 TensorFlow 会话
    with tf.Session() as sess:
        # 如果指定了 YOLO 模型的权重文件
        if yolo_weights is not None:
            # 在 'predict' 变量作用域下进行预测
            with tf.variable_scope('predict'):
                # 使用 YOLO 预测器进行模型推理，获取预测结果的张量（boxes, scores, classes）
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)

            # 载入预训练模型权重
            load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
            sess.run(load_op)

            # 进行目标检测的预测
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.size[1], image.size[0]]
                }
            )
        else:
            # 如果没有指定 YOLO 模型的权重文件，直接使用预测器进行模型推理
            boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            # 创建 TensorFlow Saver 对象，用于加载模型
            saver = tf.train.Saver()
            # 从指定路径加载模型权重
            saver.restore(sess, model_path)
            # 进行目标检测的预测
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.size[1], image.size[0]]
                }
            )

        # ---------------------------------------#
        #   Step 5:画框
        # ---------------------------------------#
        # 找到几个box，打印
        # 打印检测到的目标框数量
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # 使用指定字体创建一个 ImageFont 对象
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        # 设置绘制框的厚度
        thickness = (image.size[0] + image.size[1]) // 300

        # 遍历预测的类别及其对应的框
        for i, c in reversed(list(enumerate(out_classes))):
            # 获取预测类别的名称、框的坐标和分数
            # 获取当前预测类别的名称
            predicted_class = predictor.class_names[c]
            # 获取当前目标框的坐标信息
            box = out_boxes[i]
            # 获取当前预测的置信度分数
            score = out_scores[i]

            # 构建标签文本
            label = '{} {:.2f}'.format(predicted_class, score)

            # 创建 ImageDraw 对象用于画框框和文字
            draw = ImageDraw.Draw(image)
            # 获取文本大小，以便为文本设置合适的框
            label_size = draw.textsize(label, font)

            # 获取目标框的四个边界
            # 获取目标框的四个边界坐标
            top, left, bottom, right = box

            # 对四个边界坐标进行处理，确保不超出图像边界
            # 确保 top 不小于 0
            top = max(0, np.floor(top + 0.5).astype('int32'))
            # 确保 left 不小于 0
            left = max(0, np.floor(left + 0.5).astype('int32'))
            # 确保 bottom 不超过图像底部边界
            bottom = min(image.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
            # 确保 right 不超过图像右侧边界
            right = min(image.size[0] - 1, np.floor(right + 0.5).astype('int32'))

            # 打印目标框信息
            # 打印目标框的标签、左上角和右下角坐标
            print(label, (left, top), (right, bottom))
            # 打印目标框标签文本的大小
            print(label_size)

            # 计算文本的起始坐标
            if top - label_size[1] >= 0:
                # 如果文本上方有足够空间，文本起始坐标在目标框上方
                text_origin = np.array([left, top - label_size[1]])
            else:
                # 如果文本上方空间不足，文本起始坐标在目标框下方
                text_origin = np.array([left, top + 1])

            # 使用循环画框，创建目标框的边框
            for i in range(thickness):
                # 使用 draw.rectangle 函数创建目标框的边框
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],  # 边框的坐标范围
                    outline=predictor.colors[c]  # 边框的颜色，根据类别使用预测器中的颜色
                )

            # 创建填充框和文本
            # 使用 draw.rectangle 函数创建填充框和文本
            draw.rectangle(
                # 文本框的坐标范围
                [tuple(text_origin), tuple(text_origin + label_size)],
                # 文本框的填充颜色，根据类别使用预测器中的颜色
                fill=predictor.colors[c]
            )
            # 使用 draw.text 函数在图像上绘制目标框的标签文本
            draw.text(
                # 文本起始坐标
                text_origin,
                # 要绘制的文本内容
                label,
                # 文本的颜色（黑色）
                fill=(0, 0, 0),
                # 使用的字体对象
                font=font
            )
            # 删除 draw 对象，释放资源
            del draw

        # 显示绘制的图像
        image.show()

        # 保存绘制的结果图像
        image.save('./img/result1.jpg')


if __name__ == '__main__':

    # 当使用yolo3自带的weights的时候
    if config.pre_train_yolo3 == True:
        # 调用目标检测函数，传入图像文件路径、模型文件路径和 YOLO3 权重文件路径
        detect(config.image_file, config.model_dir, config.yolo3_weights_path)

    # 当使用模型的时候
    else:
        # 调用目标检测函数，传入图像文件路径和模型文件路径
        detect(config.image_file, config.model_dir)
