from keras.layers import Input
from frcnn import FRCNN 
from PIL import Image

frcnn = FRCNN()

while True:
    img = input('img/street1.jpg')
    try:
        image = Image.open('img/street1.jpg')
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
frcnn.close_session()
    
