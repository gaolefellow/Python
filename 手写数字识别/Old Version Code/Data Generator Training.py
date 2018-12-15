import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import transform,filters,io
import os


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# for i in range(len(x_train)):
#     rescale = transform.resize(x_train[i],(224,224))
#     io.imsave("./data/x_train_png/"+str(i)+'.png',rescale)
#
# for i in range(len(x_test)):
#     rescale = transform.resize(x_test[i], (224, 224))
#     io.imsave("./data/x_test_png/" + str(i) + '.png',rescale)

def get_train_batch(x_train,y_train,batch_size):
    num = len(x_train)
    while 1:
        X = []
        Y = []
        count = 0
        for i in range(num):
            x = x_train[i]
            x = transform.resize(x,(224,224))
            threshold = filters.threshold_yen(x)
            x = np.reshape(x, (224, 224, 1))
            X.append(x)
            Y.append(y_train[i])
            count += 1
            if count == batch_size:
                yield np.array(X), np.array(Y)
                count = 0
                X = []
                Y = []

def get_train_flow(path,label,batch_size):
    img = os.listdir(path)
    num = len(img)
    while 1:
        X = []
        Y = []
        count = 0
        for i in range(num):
            x = io.imread(path+str(i)+'.png')
            threshold = filters.threshold_yen(x)
            x = (x > threshold)
            x = np.reshape(x,(224,224,1))
            X.append(x)
            Y.append(label[i])
            count += 1
            if count == batch_size:
                yield np.array(X),np.array(Y)
                count = 0
                X = []
                Y = []

def get_test_flow(path,label):
    img = os.listdir(path)
    num = len(img)
    while 1:
        for i in range(num):
            x = io.imread(path+str(i)+'.png')
            threshold = filters.threshold_yen(x)
            x = (x > threshold)
            x = np.reshape(x, (224, 224, 1))
            x = np.expand_dims(x,axis=0)
            y = label[i]
            y = np.expand_dims(y,axis=0)
            yield (x , y)

# test = get_test_flow('./data/x_test_png/',y_test)
# for i in test:
#     print(np.shape(i[1]))
#     break
#
# train = get_train_flow('./data/x_train_png/',y_train,32)
# for i in train:
#     print(np.shape(i[1]))
#     break



def Train_Model_VGG_self():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (7, 7), activation='relu', strides=(1, 1),padding='same', input_shape=(224, 224, 1)),
        tf.keras.layers.Conv2D(32, (7, 7), activation='relu',strides=(1, 1),padding='same', ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', strides=(1, 1), padding='same', ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu',strides=(1, 1),padding='same', ),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same', ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu',strides=(1, 1),padding='same', ),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', strides=(2, 2), padding='same', ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    opt = tf.keras.optimizers.Adadelta()
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit_generator(get_train_flow('./data/x_train_png/',y_train,10),steps_per_epoch=6000,epochs=2)
    model.save('./model/VGG_self_224.h5')
    return model

model = Train_Model_VGG_self()
model = tf.keras.models.load_model('./model/VGG_self_224.h5')

print(model.evaluate_generator(get_test_flow('./data/x_test_png/',y_test),steps=10000))











