import numpy as np
import utils
import cv2
from keras import backend as K
from Alexnet import Alexnet

#通道放到最后
K.image_data_format() == 'channel_last'


if __name__ == '__main__':
    model = Alexnet()
    model.load_weights("./last1.h5") #训练好的模型
    #测试图片，并处理
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor, axis=0)
    img_resize = utils.resize_image(img_nor, (224, 224))
    # utils.print_answer(np.argmax(model.predict(img)))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo", img)
    cv2.waitKey(0)