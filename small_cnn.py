import tensorflow as tf
import keras
from keras import Sequential

from keras import datasets, layers, models

def get_model(img_height,img_width,num_classes,lr):
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    return model