from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout

def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    model = Sequential()

    # L1 输入图像特征提取 48 通道
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding="valid",
            input_shape=input_shape,
            activation="relu"
        )
    )

    # L1数据归一化
    model.add(BatchNormalization())

    # L1最大池化
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        ))

    # L2 卷积 特征128
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            activation='relu'
        )
    )

    # L2 归一化
    model.add(BatchNormalization())

    # L2 池化
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # L3 卷积
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # L4 卷积
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            activation='relu'
        )
    )

    # L5
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
        )
    )

    # 池化 改变w h
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # L6
    # model.add(
    #     Conv2D(
    #         filters=128,
    #         kernel_size=(3, 3),
    #         strides=(2, 2),
    #         padding='valid',
    #         activation='relu',
    #     )
    # )

    # 最后连接全连接层 两个全连接层
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_shape, activation='softmax'))

    return model

