from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.preprocessing import image
from keras import backend as K
from sklearn.metrics import classification_report
import numpy as np
import keras
import sys
import os
import json

from model import MODELS
from model.utils import get_input_shape, summarize_accuracy

def files_in_directory(path: str):
    """
    Recursively count files in a directory
    """
    count = 0
    for _, _, files in os.walk(path):
        count += len(files)
    
    return count

def train(weights_output="deep_network.h5"):
    with open('config.json') as json_file:
        config = json.load(json_file)

    # dimensions of our images.
    img_width, img_height = config["input_width"], config["input_height"]

    train_data_dir = config["training_dir"]
    validation_data_dir = config["validation_dir"]
    nb_train_samples = files_in_directory(train_data_dir)
    nb_validation_samples = files_in_directory(validation_data_dir)
    epochs = config["train_epochs"]
    batch_size = config["batch_size"]
    input_shape = get_input_shape(img_width, img_height)
    patch_size = tuple(config["patch_size"])
    patch_shape = get_input_shape(patch_size[0], patch_size[1])

    classes = sorted(config["classes"])
    num_classes = len(classes)

    model_name = config["model_name"]
    # Check if the name is one of the supported custom models
    # TODO: Add support for prebuilt Keras models
    if model_name not in MODELS:
        raise KeyError("Model name not recognized please try one of the following: " + str(list(MODELS.keys())))

    model = MODELS[model_name](input_shape=input_shape, classes=num_classes)

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=[categorical_accuracy, "accuracy"]
    )

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes
    )

    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        workers=4
    )

    model.save_weights(weights_output)

    # Summarize accuracy for each class in the validation dataset
    summarize_accuracy(model, classes, validation_data_dir, img_width, img_height)

if __name__ == "__main__":
    if(len(sys.argv) > 1):
        print("Trained model will be saved as:", sys.argv[1])
        train(sys.argv[1])
    else:
        train()
