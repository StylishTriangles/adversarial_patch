from keras import Model
from keras import backend as K
from keras.preprocessing import image
import numpy as np

import os

# Assumes 3 channels in input
def get_input_shape(img_width: int, img_height: int):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    return input_shape

def summarize_accuracy(model: Model, classes: list, validation_dir: str, img_width: int, img_height: int):
    """
    Runs predictions on all validation data and displays accuracy for each class.
    """
    num_classes = len(classes)
    ans = dict(zip(classes, [0]*num_classes))
    total = dict(zip(classes, [0]*num_classes))
    directory = validation_dir
    for dirname in os.listdir(directory):
        if dirname not in classes:
            print(dirname, "directory skipped, not in classes list")
            continue
        for filename in os.listdir(os.path.join(directory, dirname)):
            img = image.load_img(os.path.join(directory, dirname, filename), target_size=(img_width, img_height))
            y = image.img_to_array(img)
            y = np.expand_dims(y, axis=0)
            output = model.predict(y)[0]

            total[dirname] += 1
            for classname, value in zip(classes, output):
                if classname == dirname and value >= max(output):
                    ans[dirname]+=1
    for classname in classes:
        rate = 1
        if total[classname] != 0:
            rate = ans[classname]/total[classname]
        print(
            classname, 
            "\tcount:", total[classname],
            "\tsuccesses:", ans[classname], 
            "\trate:", rate
        )