import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import transform,filters,io
import os
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train,(len(x_train),28,28,1))
x_test = np.reshape(x_test,(len(x_test),28,28,1))
x_train = np.array(x_train,dtype=np.float)
x_test = np.array(x_test,dtype=np.float)
x_train /= 255.
x_test /=255.


def Train_Model_Inception(x_train,y_train):
    img_input = layers.Input((28,28,1))
    x = layers.Conv2D(32,(3,3),padding='same')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32,(3,3),strides=(2,2),padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    pathway1 = layers.Conv2D(192,(1,1),padding='same')(x)
    pathway1 = layers.BatchNormalization()(pathway1)
    pathway1 = layers.Activation('relu')(pathway1)
    pathway2 = layers.Conv2D(128,(1,1),padding='same')(x)
    pathway2 = layers.BatchNormalization()(pathway2)
    pathway2 = layers.Activation('relu')(pathway2)
    pathway2 = layers.Conv2D(160,(1,7),padding='same')(pathway2)
    pathway2 = layers.BatchNormalization()(pathway2)
    pathway2 = layers.Activation('relu')(pathway2)
    pathway2 = layers.Conv2D(192,(7,1),padding='same')(pathway2)
    pathway2 = layers.BatchNormalization()(pathway2)
    pathway2 = layers.Activation('relu')(pathway2)
    x = layers.concatenate([pathway1,pathway2],axis=3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024,activation='relu')(x)
    x = layers.Dense(10,activation='softmax')(x)

    model = tf.keras.models.Model(img_input,x)

    opt = tf.keras.optimizers.Adadelta()
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=5,validation_split=0.1)
    model.save('./model/Inception1.h5')
    return model

def Train_Model_Inception2(x_train,y_train):
    img_input = layers.Input((28,28,1))
    x = layers.Conv2D(32,(3,3),padding='same')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.Conv2D(32,(3,3),strides=(2,2),padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.Conv2D(64,(3,3),padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    pathway1 = layers.AveragePooling2D((3, 3),strides=(1, 1),padding='same')(x)
    pathway1 = layers.Conv2D(128,(1,1),padding='same')(pathway1)
    pathway1 = layers.BatchNormalization()(pathway1)
    pathway1 = layers.Activation('elu')(pathway1)
    pathway2 = layers.Conv2D(384,(1,1),padding='same')(x)
    pathway2 = layers.BatchNormalization()(pathway2)
    pathway2 = layers.Activation('elu')(pathway2)
    pathway3 = layers.Conv2D(192,(1,1),padding='same')(x)
    pathway3 = layers.BatchNormalization()(pathway3)
    pathway3 = layers.Activation('elu')(pathway3)
    pathway3 = layers.Conv2D(224,(7,1),padding='same')(pathway3)
    pathway3 = layers.BatchNormalization()(pathway3)
    pathway3 = layers.Activation('elu')(pathway3)
    pathway3 = layers.Conv2D(256,(1,7),padding='same')(pathway3)
    pathway3 = layers.BatchNormalization()(pathway3)
    pathway3 = layers.Activation('elu')(pathway3)
    pathway4 = layers.Conv2D(192,(1,1),padding='same')(x)
    pathway4 = layers.BatchNormalization()(pathway4)
    pathway4 = layers.Activation('elu')(pathway4)
    pathway4 = layers.Conv2D(192,(1,7),padding='same')(pathway4)
    pathway4 = layers.BatchNormalization()(pathway4)
    pathway4 = layers.Activation('elu')(pathway4)
    pathway4 = layers.Conv2D(224,(7,1),padding='same')(pathway4)
    pathway4 = layers.BatchNormalization()(pathway4)
    pathway4 = layers.Activation('elu')(pathway4)
    pathway4 = layers.Conv2D(224,(1,7),padding='same')(pathway4)
    pathway4 = layers.BatchNormalization()(pathway4)
    pathway4 = layers.Activation('elu')(pathway4)
    pathway4 = layers.Conv2D(256,(7,1),padding='same')(pathway4)
    pathway4 = layers.BatchNormalization()(pathway4)
    pathway4 = layers.Activation('elu')(pathway4)
    x = layers.concatenate([pathway1,pathway2,pathway3,pathway4],axis=3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10,activation='softmax')(x)

    model = tf.keras.models.Model(img_input,x)

    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=10,validation_split=0.1,
                  callbacks=[TensorBoard(log_dir='./temp/log'),EarlyStopping()])
    model.save('./model/Inception1_10_14.h5')
    return model


model = Train_Model_Inception2(x_train,y_train)
print(model.evaluate(x_test,y_test))