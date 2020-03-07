# constants used for adversarial patching
from keras import applications
from keras import backend as K

import numpy as np
import os
import sys
import json

this_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.abspath(os.path.join(this_dir, ".."))) # add the main directory of adversarial_patch
from model import MODELS

# Assumes 3 channels in input
def get_input_shape(img_width: int, img_height: int):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    return input_shape

with open('config.json') as json_file:
    config = json.load(json_file)

INPUT_WIDTH, INPUT_HEIGHT = config["input_width"], config["input_height"]
LABELS = sorted(config["classes"])
NUM_LABELS = len(LABELS)

def name_to_label(name: str) -> int:
    return LABELS.index(name)


TARGET_LABEL = name_to_label(config["target_class"])
PATCH_SIZE = tuple(config["patch_size"])
PATCH_SHAPE = get_input_shape(PATCH_SIZE[0], PATCH_SIZE[1])
BATCH_SIZE = config["batch_size"]

INPUT_SIZE = (INPUT_WIDTH, INPUT_HEIGHT)
INPUT_SHAPE = get_input_shape(INPUT_WIDTH, INPUT_HEIGHT)


# Ensemble of models
NAME_TO_MODEL = {
    'xception': applications.xception.Xception,
    'vgg16': applications.vgg16.VGG16,
    'vgg19': applications.vgg19.VGG19,
    'resnet50': applications.resnet50.ResNet50,
    'inceptionv3': applications.inception_v3.InceptionV3,
}
NAME_TO_MODEL.update(MODELS)

MODEL_NAMES = sorted(list(NAME_TO_MODEL.keys()))

# Data augmentation
# Empirically found that training with a very wide scale range works well
# as a default
SCALE_MIN = 0.3
SCALE_MAX = 1.5
ROTATE_MAX = np.pi/8 # 22.5 degrees in either direction

MAX_ROTATION = 22.5

# Local data dir to write files to
DATA_DIR = config["output_dir"]

# validation images set
VALIDATION_DIR = config["validation_dir"]
