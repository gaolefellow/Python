import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import transform,filters,io
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train,(len(x_train),28,28,1))
x_test = np.reshape(x_test,(len(x_test),28,28,1))
x_train = np.array(x_train,dtype=np.float)
x_test = np.array(x_test,dtype=np.float)
x_train /= 255.
x_test /=255.


print(x_train[0])



def Train_Model_VGG_224(x_train,y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', strides=(1, 1), padding='same',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10,activation='softmax')
    ])
    opt = tf.keras.optimizers.Adadelta()
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=5)
    model.save('./model/VGG_self_10_2.h5')
    return model

model = Train_Model_VGG_224(x_train,y_train)
model = tf.keras.models.load_model('./model/VGG_self_10_2.h5')
print(model.evaluate(x_test,y_test))