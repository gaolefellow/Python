import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import transform,filters,io
import os
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint



class Model():
    def __init__(self):
        super(Model, self).__init__()
        self.input_size = (28,28,1)
        self.optimize = tf.keras.optimizers.Adam()
        self.earlystop = EarlyStopping(monitor='val_acc',patience=3)


    def conv2d_bn(self,x, filters, kernel, strides=(1, 1), padding='same'):
        x = layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        return x

    def inception_c(self,x):
        branch1 = layers.AveragePooling2D(pool_size=(3,3),strides=1,padding='same')(x)

        branch2 = self.conv2d_bn(x,256,(1,1))

        branch3 = self.conv2d_bn(x,384,(1,1))
        branch31 = self.conv2d_bn(branch3,256,(1,3))
        branch32 = self.conv2d_bn(branch3,256,(3,1))
        branch3 = layers.concatenate([branch31,branch32],axis=3)

        branch4 = self.conv2d_bn(x,384,1)
        branch4 = self.conv2d_bn(branch4,448,(1,3))
        branch4 = self.conv2d_bn(branch4,512,(3,1))
        branch41 = self.conv2d_bn(branch4,256,(3,1))
        branch42 = self.conv2d_bn(branch4,256,(1,3))
        branch4 = layers.concatenate([branch41,branch42],axis=3)

        x = layers.concatenate([branch1,branch2,branch3,branch4],axis=3)
        return x

    def Net(self,input_size):
        img = layers.Input(input_size) #28*28*1

        x = self.conv2d_bn(img,16,3)
        x = self.conv2d_bn(x,32,3)
        x = layers.MaxPooling2D()(x) #14*14*1
        x = self.conv2d_bn(x,64,3)
        x = self.conv2d_bn(x,64,3,2) #7*7*1
        x = self.inception_c(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.8)(x)
        x = layers.Dense(10,activation='softmax')(x)

        model = tf.keras.models.Model(img,x)

        return model

    def Train(self,data,label):
        model = self.Net(self.input_size)
        model.compile(optimizer=self.optimize,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        model.fit(data, label, epochs=30, validation_split=0.1,
                  callbacks=[TensorBoard(log_dir='./log'),
                             EarlyStopping(monitor='val_loss',patience=3),
                             ModelCheckpoint('./model/Last_Version.h5',monitor='val_acc',save_best_only=True,mode='max')])
        return model



if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train,(len(x_train),28,28,1))
    x_test = np.reshape(x_test,(len(x_test),28,28,1))
    x_train = np.array(x_train,dtype=np.float)
    x_test = np.array(x_test,dtype=np.float)
    x_train /= 255.
    x_test /=255.

    OurModel = Model()
    OurModel.Train(x_train,y_train)








