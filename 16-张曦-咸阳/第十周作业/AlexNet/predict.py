from AlexNet import AlexNet
import cv2
import numpy as np
import utils

if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./logs/last1.h5")
    img = cv2.imread("xiu.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    img_nor = np.expand_dims(img, axis=0)
    img_resize = utils.resize_image(img_nor, (224, 224))

    print(utils.print_answer(np.argmax(model.predict(img_resize))))

    cv2.imshow("ooo", img)
    cv2.waitKey(0)