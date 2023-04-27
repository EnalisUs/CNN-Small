import matplotlib.pyplot as plt
import numpy as np
import os
import keras
import tensorflow as tf
from keras import Sequential

from keras import datasets, layers, models


def augment_using_ops(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.rot90(images)
    return images, labels


batch_size = 64
img_height = 32
img_width = 32
data_dir = "./data"
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(500).map(augment_using_ops, num_parallel_calls=AUTOTUNE) \
    .prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_ds_val = val_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds_val))
first_image = image_batch[0]
print(first_image)
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
num_classes = len(class_names)

model = Sequential([
    layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                  input_shape=(img_height, img_width, 3)),
    layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.4),
    layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),
    layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    layers.GlobalAvgPool2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu',
                 kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                 bias_regularizer=keras.regularizers.L2(1e-4),
                 activity_regularizer=keras.regularizers.L2(1e-5)),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()
epochs = 250;
history = model.fit(
    normalized_ds,
    validation_data=normalized_ds_val,
    epochs=epochs,
    verbose=1
)
model.save('small.h5')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

