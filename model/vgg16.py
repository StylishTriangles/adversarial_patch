from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization

from .utils import get_input_shape

class SimpleVGG16(Sequential):
    """
    SimpleVGG16 is a simplified version of a VGG16A model.
    The changes include input shape reduction (to 128x128 px) and smaller convolutional and dense layers
    """
    def __init__(self, input_shape = None, classes = 10):
        super(SimpleVGG16, self).__init__()

        if input_shape is None:
            input_shape = get_input_shape(128, 128)

        kernel_size = (3,3)

        self.add(Conv2D(32, kernel_size, input_shape=input_shape, activation='relu'))
        self.add(Dropout(0.2))
        self.add(MaxPooling2D(pool_size=2))

        self.add(Conv2D(64, kernel_size, activation='relu'))
        self.add(MaxPooling2D(pool_size=2))
        self.add(Dropout(0.2))

        self.add(Conv2D(128, kernel_size, activation='relu'))
        self.add(Dropout(0.2))
        self.add(Conv2D(128, kernel_size, activation='relu'))
        self.add(MaxPooling2D(pool_size=2))
        self.add(Dropout(0.2))

        self.add(Conv2D(256, kernel_size, activation='relu'))
        self.add(Dropout(0.2))
        self.add(Conv2D(256, kernel_size, activation='relu'))
        self.add(MaxPooling2D(pool_size=2))
        self.add(Dropout(0.2))

        self.add(Flatten())
        self.add(Dense(512, activation='relu'))
        self.add(Dropout(0.2))
        self.add(BatchNormalization())
        self.add(Dense(256, activation='tanh'))
        self.add(Dropout(0.2))
        self.add(BatchNormalization())
        self.add(Dense(classes, activation="softmax"))