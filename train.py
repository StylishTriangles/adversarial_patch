from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.preprocessing import image
from keras import backend as K
from sklearn.metrics import classification_report
import numpy as np
import keras
import sys
import os

import mod

# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'data'
validation_data_dir = 'validation'
nb_train_samples = 8800
nb_validation_samples = 800
epochs = 800
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# model = Sequential()

# model.add(Conv2D(16, kernel_size, input_shape=input_shape, activation='relu'))
# model.add(Conv2D(16, kernel_size, activation='relu'))
# model.add(Conv2D(16, kernel_size, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))

# model.add(Conv2D(64, kernel_size, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))

# model.add(Conv2D(128, kernel_size, activation='relu'))

# model.add(Dropout(0.1))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(10, activation="softmax"))


##### MY STUFF


classes = [
    "afterburner",  "apple",  "banana", "cat", "mug", "orange",  "racecar",  "slav",  "stephen",  "wagon"
]
num_models = len(classes)
model = mod.make_model(input_shape, num_models)


#### WOLOLO
# model = Sequential()
# model.add(Dense(256, activation='relu', input_shape=input_shape))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))


# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

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
# only rescaling
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    workers=4
)

model.save_weights('cool_net.h5')


ans = dict(zip(classes, [0]*num_models))
total = dict(zip(classes, [0]*num_models))
directory = 'validation'
for dirname in os.listdir(directory):
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
    print(
        classname, 
        "\tcount:", total[classname],
        "\tsuccesses:", ans[classname], 
        "\trate:", ans[classname]/total[classname]
    ) 