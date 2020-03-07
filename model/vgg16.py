from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, InputLayer
from keras import backend as K

from .utils import get_input_shape

class SimpleVGG16(Sequential):
    """
    SimpleVGG16 is a simplified version of a VGG16A model.
    The changes include input shape reduction (to 128x128 px) and smaller convolutional and dense layers
    """
    def __init__(self, input_shape = None, input_tensor = None, classes = 10):
        super(SimpleVGG16, self).__init__()

        kernel_size = (3,3)

        if input_shape is None:
            input_shape = get_input_shape(128, 128)

        if input_tensor is None:
            img_input = InputLayer(input_shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = InputLayer(input_tensor=input_tensor, input_shape=input_shape)
            else:
                img_input = input_tensor

        self.add(img_input)
        self.add(Conv2D(32, kernel_size, activation='relu'))
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