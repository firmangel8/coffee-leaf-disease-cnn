import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = "data_class_generated"
image_size = 224

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0,
                                                             shear_range=0.2,
                                                             zoom_range=0.2,
                                                             width_shift_range=0.2,
                                                             height_shift_range=0.2,
                                                             fill_mode="nearest")
batch_size = 32
train_data = train_datagen.flow_from_directory(os.path.join(base_dir, "train"),
                                               target_size=(
                                                   image_size, image_size),
                                               batch_size=batch_size,
                                               class_mode="categorical"
                                               )
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
test_data = test_datagen.flow_from_directory(os.path.join(base_dir, "test"),
                                             target_size=(
                                                 image_size, image_size),
                                             batch_size=batch_size,
                                             class_mode="categorical"
                                             )

categories = list(train_data.class_indices.keys())
print(categories)

base_model = keras.applications.MobileNet(
    weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))


base_model.trainable = False
inputs = keras.Input(shape=(image_size, image_size, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(len(categories), activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=x,
                    name="LeafDisease_MobileNet")

model.summary()
optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(
    from_logits=True), metrics=[keras.metrics.CategoricalAccuracy()])

history = model.fit_generator(train_data,
                              validation_data=test_data,
                              epochs=25,
                              steps_per_epoch=int(1028 / batch_size),
                              validation_steps=int(296 / batch_size)
                              )


model.evaluate(test_data)
model.save('leaf-coffee-cnn-mobilenet.h5')
