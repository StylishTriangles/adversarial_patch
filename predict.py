from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras import backend as K
from typing import List, Tuple
import keras
import sys
import os
import json
import numpy as np

from model.utils import get_input_shape, summarize_accuracy
from model import MODELS


if __name__ == "__main__":
    with open('config.json') as json_file:
        config = json.load(json_file)

    classes = config["classes"]
    num_classes = len(classes)

    # dimensions of our images.
    img_width, img_height = config["input_width"], config["input_height"]
    input_shape = get_input_shape(img_width, img_height)

    model_name = config["model_name"]
    # Check if the name is one of the supported custom models
    # TODO: Add support for prebuilt Keras models
    if model_name not in MODELS:
        raise KeyError("Model name not recognized please try one of the following: " + str(list(MODELS.keys())))

    model = MODELS[model_name](input_shape=input_shape, classes=num_classes)

    model.load_weights(sys.argv[1])

    if len(sys.argv) > 2:
        img = image.load_img(sys.argv[2], target_size=(img_width, img_height))

        y = image.img_to_array(img)
        y = np.expand_dims(y, axis=0)

        output = model.predict(y)[0]

        print(sorted(list(zip(output,classes)), reverse=True))
    else:
        # Run predictions on all validation data and display accuracy
        summarize_accuracy(model, classes, config["validation_dir"], img_width, img_height)
