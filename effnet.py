import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import cv2

# from IPython.display import Image
import tensorflow as tf


import tensorflow.keras as keras
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications import EfficientNetB2
# import efficientnet.keras as efn
import tensorflow.keras.applications.efficientnet as efn
# model = efn.EfficientNetB0(weights='imagenet')
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input

# import segmentation_models as sm
# sm.set_framework('tf.keras')
# sm.framework()


train_DIR = "data_class_generated/train/"
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   vertical_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_DIR,
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    target_size=(250, 250))

test_DIR = "data_class_generated/test/"
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)


validation_generator = validation_datagen.flow_from_directory(test_DIR,
                                                              batch_size=128,
                                                              class_mode='categorical',
                                                              target_size=(250, 250))

print(validation_generator.class_indices)
class2index = validation_generator.class_indices

index2class = {v: k for k, v in class2index.items()}
print(index2class)

efficientNet = efn.EfficientNetB1(weights='imagenet')
last_output = efficientNet.layers[-1].output

# x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)(last_output)
# x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(6, activation=tf.nn.softmax)(x)

model = tf.keras.Model(efficientNet.input, x)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=1,
                                            verbose=1,
                                            factor=0.25,
                                            min_lr=0.000003)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator,
                    callbacks=[learning_rate_reduction])
