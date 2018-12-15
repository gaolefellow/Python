import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import cv2
from skimage import transform,filters

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def data_process(x_train,x_test):
    #放大数据集
    x_train = [transform.resize(image,(112,112)) for image in x_train]
    x_test = [transform.resize(image,(112,112)) for image in x_test]

    #数据集二值化
    threshold = filters.threshold_otsu(x_train[0])
    x_train = [(image > threshold) for image in x_train]
    x_test = [(image > threshold) for image in x_test]

    #将数据集变成3维
    x_train = np.reshape(np.array(x_train),(60000,112,112,1))
    x_test = np.reshape(np.array(x_test),(10000,112,112,1))

    np.save('./data/x_train.npy',x_train)
    np.save('./data/x_test.npy',x_test)

# data_process(x_train,x_test)

x_train = np.load('./data/x_train.npy')
x_test = np.load('./data/x_test.npy')

def train_model_CNN3(x_train,y_train):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5, 5), activation='relu', strides=(1, 1), input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(1, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)
        model.save('./model/CNN3.h5')


def Train_Model_CNN4(x_train,y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',strides=(1,1),input_shape=(112,112,1)),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(128,(5,5),activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=4)
    model.save('./model/CNN5.h5')


def Train_Model_VGG_self(x_train,y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=(1, 1),padding='same', input_shape=(112, 112, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',strides=(1, 1),padding='same', ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu',strides=(1, 1),padding='same', ),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu',strides=(1, 1),padding='same', ),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=5)
    model.save('./model/VGG_self.h5')
    return model

# Train_Model_CNN4(x_train,y_train)
# model = tf.keras.models.load_model('./model/CNN5.h5')

model = Train_Model_VGG_self(x_train,y_train)
print(model.evaluate(x_test,y_test))



img = np.reshape(x_test[3]*255,(112,112))
print(img)
cv2.imwrite('ref.png',img)



