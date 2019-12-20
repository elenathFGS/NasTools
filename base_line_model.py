#    (\~---.
#    /   (\-`-/)
#   (      ' ' )
#    \ (  \_Y_/\
#     ""\ \___//
#        `w   " 
# by Elenath Feng
# keras implementation for the baseline model and test models in the paper
# <Convolutional neural networks at constrained time cost> He at.al

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, ZeroPadding2D
from keras_extra_Layers.SpatialPyramidPooling import SpatialPyramidPooling


def BaselineModel():
    BaselineModel = Sequential()
    BaselineModel.add(Conv2D(filters=64, kernel_size=7, strides=2,
                             input_shape=(224, 224, 3), activation='relu'))
    BaselineModel.add(MaxPool2D(pool_size=(3, 3), strides=3))
    BaselineModel.add(ZeroPadding2D(padding=(2, 2)))
    BaselineModel.add(Conv2D(filters=128, kernel_size=5, activation='relu'))
    BaselineModel.add(MaxPool2D(pool_size=(2, 2), strides=2))

    for i in range(3):
        BaselineModel.add(ZeroPadding2D(padding=(1, 1)))
        BaselineModel.add(Conv2D(filters=256, kernel_size=3, activation='relu'))

    BaselineModel.add(SpatialPyramidPooling([1, 2, 4]))
    BaselineModel.add(Dense(4096, activation='relu'))
    BaselineModel.add(Dense(4096, activation='relu'))
    BaselineModel.add(Dense(1000, activation='relu'))

    return BaselineModel

def BaselineModel_B():
    BaselineModel = Sequential()
    BaselineModel.add(Conv2D(filters=64, kernel_size=7, strides=2,
                             input_shape=(224, 224, 3), activation='relu'))
    BaselineModel.add(MaxPool2D(pool_size=(3, 3), strides=3))
    BaselineModel.add(ZeroPadding2D(padding=(2, 2)))
    BaselineModel.add(Conv2D(filters=128, kernel_size=5, activation='relu'))
    BaselineModel.add(MaxPool2D(pool_size=(2, 2), strides=2))

    padding = 0
    for i in range(6):
        if padding % 2 == 1:
            BaselineModel.add(ZeroPadding2D(padding=(1, 1)))
            # For two sequential 2*2 filters, we do not pad the first and pad 1 pixel on the second
        padding += 1
        BaselineModel.add(Conv2D(filters=256, kernel_size=2, activation='relu'))

    BaselineModel.add(SpatialPyramidPooling([1, 2, 4]))
    BaselineModel.add(Dense(4096, activation='relu'))
    BaselineModel.add(Dense(4096, activation='relu'))
    BaselineModel.add(Dense(1000, activation='relu'))
    return BaselineModel

def BaselineModel_C():
    BaselineModel = Sequential()
    BaselineModel.add(Conv2D(filters=64, kernel_size=7, strides=2,
                             input_shape=(224, 224, 3), activation='relu'))
    BaselineModel.add(MaxPool2D(pool_size=(3, 3), strides=3))

    BaselineModel.add(ZeroPadding2D(padding=(1, 1)))
    BaselineModel.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    BaselineModel.add(ZeroPadding2D(padding=(1, 1)))
    BaselineModel.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    BaselineModel.add(MaxPool2D(pool_size=(2, 2), strides=2))
    for i in range(3):
        BaselineModel.add(ZeroPadding2D(padding=(1, 1)))
        BaselineModel.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
    BaselineModel.add(SpatialPyramidPooling([1, 2, 4]))
    BaselineModel.add(Dense(4096, activation='relu'))
    BaselineModel.add(Dense(4096, activation='relu'))
    BaselineModel.add(Dense(1000, activation='relu'))
    return BaselineModel
