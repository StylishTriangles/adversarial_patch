from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.constraints import maxnorm

from .utils import get_input_shape

class CIFAR10(Sequential):
    def __init__(self, input_shape = None, classes = 10):
        super(CIFAR10, self).__init__()

        if input_shape is None:
            input_shape = get_input_shape(64, 64)

        self.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
        self.add(Dropout(0.2))
        self.add(BatchNormalization())

        self.add(Conv2D(64, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.2))
        self.add(BatchNormalization())

        self.add(Conv2D(64, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.2))
        self.add(BatchNormalization())

        self.add(Conv2D(128, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.2))
        self.add(BatchNormalization())

        self.add(Flatten())
        self.add(Dropout(0.2))

        self.add(Dense(256, kernel_constraint=maxnorm(3)))
        self.add(Activation('relu'))
        self.add(Dropout(0.2))
        self.add(BatchNormalization())
        self.add(Dense(128, kernel_constraint=maxnorm(3)))
        self.add(Activation('relu'))
        self.add(Dropout(0.2))
        self.add(BatchNormalization())
        self.add(Dense(classes))
        self.add(Activation('softmax'))
