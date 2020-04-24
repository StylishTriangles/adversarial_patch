from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import keras
import sys
import os
import json
import random

from model import MODELS
from model.utils import get_input_shape, summarize_accuracy


class BalancedImageFlow:
    def __init__(
        self, 
        image_generator: ImageDataGenerator,
        directory: str, 
        classes: list,
        target_size: tuple = (256,256), 
        batch_size: int = 32, 
        class_mode: str = "categorical"
        ):
        self.flow_generator = image_generator.flow_from_directory(
            directory=directory,
            classes=classes, 
            target_size=target_size, 
            batch_size=1, # take the next image and discard if class appeared too many times
            class_mode=class_mode
        )
        self.batch_size = batch_size
        self.classes = classes
        self.test = False
        # how many instances of each class appeared so far
        self.counters = np.ones(len(classes))
        # total classes
        self.total = len(classes)
        # the expected percentage of instances of given class
        self.expected_ratio = len(self.classes)/self.total

    def __iter__(self):
        return self

    def _should_discard(self, labels: np.array):
        """Should the set be discarded?"""
        nz_index = labels.nonzero()[0][0]
        chance_to_delete = (self.counters[nz_index]/self.total - self.expected_ratio)*len(self.classes)
        if random.random() < chance_to_delete:
            return True
        return False

    def __next__(self):
        img_batch = []
        labels_batch = []
        while len(img_batch) < self.batch_size:
            images, labels = next(self.flow_generator)
            if not self._should_discard(labels):
                img_batch.extend(images)
                labels_batch.extend(labels)
                self.counters += labels[0] # labels is a 2D array with labels array at index 0
                self.total += 1
        # the labels returned is a 2 dimensional array with just
        return (np.array(img_batch), np.array(labels_batch))


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
        rotation_range=20,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    test_datagen = ImageDataGenerator()

    train_generator = BalancedImageFlow(
        test_datagen,
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
