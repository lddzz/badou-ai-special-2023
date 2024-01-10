from fast_rcnn import Fast_RCNN
from PIL import Image
import numpy as np
import cv2

FRCNN_model = Fast_RCNN()

# 调用摄像头
capture = cv2.VideoCapture(0)

while True:
    # 读取某一帧
    ref, frame = capture.read()
    # BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Image变为unit8格式
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = np.array(FRCNN_model.detect_image(frame))
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("video", frame)
    # 等待30毫秒，无按键继续，有按键返回按键
    c = cv2.waitKey(30) & 0xff
    # 按键是Esc退出循环，结束程序
    if c == 27:
        capture.release()
        break

Fast_RCNN.close_session()
