from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras import backend as K
import keras
import sys
import os
import numpy as np

import mod


classes = [
    "afterburner",  "apple",  "banana", "cat", "mug", "orange",  "racecar",  "slav",  "stephen",  "wagon"
]

num_models = len(classes)

# dimensions of our images.
img_width, img_height = 128, 128

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = mod.make_model(input_shape, num_models)

model.load_weights(sys.argv[1])


# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

if len(sys.argv) > 2:
    img = image.load_img(sys.argv[2], target_size=(img_width, img_height))

    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)

    output = model.predict(y)[0]

    print(sorted(list(zip(output,classes)), reverse=True))
else:
    ans = dict(zip(classes, [0]*num_models))
    total = dict(zip(classes, [0]*num_models))
    macks = 0.0
    directory = 'data'
    for dirname in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, dirname)):
            img = image.load_img(os.path.join(directory, dirname, filename), target_size=(img_width, img_height))
            y = image.img_to_array(img)
            y = np.expand_dims(y, axis=0)
            output = model.predict(y)[0]

            total[dirname] += 1
            for classname, value in zip(classes, output):
                if classname == dirname and value >= max(output)-0.1:
                    ans[dirname]+=1
                    if classname == "stephen":
                        if value > macks:
                            macks = value
                            bestest = filename
    for classname in classes:
        print(
            classname, 
            "\tcount:", total[classname],
            "\tsuccesses:", ans[classname], 
            "\trate:", ans[classname]/total[classname]
        ) 
    # train_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True)
    # validation_generator = train_datagen.flow_from_directory(
    #     "validation",
    #     target_size=(img_width, img_height),
    #     batch_size=1,
    #     class_mode='categorical'
    # )
    # y_pred = model.predict_classes(validation_generator)
    # print(y_pred)