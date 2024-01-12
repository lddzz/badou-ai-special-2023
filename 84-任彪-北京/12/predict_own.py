from keras.layers import Input
from FRCNN_own import FRCNN
from PIL import Image


frcnn = FRCNN()

try:
    image = Image.open('test.jpeg')
except:
    print('Open Error! Try again!')
else:
    r_image = frcnn.detect_image(image)
    r_image.show()
frcnn.close_session()
