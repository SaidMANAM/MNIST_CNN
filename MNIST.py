from typing import Any, Union

import pandas as pd
import numpy as np
from numpy.core._multiarray_umath import ndarray
from pandas import DataFrame, Series
from pandas.core.arrays import ExtensionArray

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau


train = pd.read_csv("/home/said/MNIST/train.csv")
test = pd.read_csv("/home/said/MNIST/test.csv")
#getting the labels separated from the training data
Y_train= train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)



#print("Y_train",Y_train)
#print("X_train",X_train)
#print("train",train)
#print("test",test)


# Normalizing the data
X_train = X_train / 255.0
test = test / 255.0
#print("X_train: normalized:",X_train)

#print("test: normalized:",test)
#transforming the data into the shape supported by the methods used in the model
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
#print("X_train: reshaped:",X_train)

#print("test: reshaped:",test)
Y_train = to_categorical(Y_train, num_classes = 10)
#print("Y_train",Y_train)
#free space because train will not be used anymore
del train

#Splitting data training data into training and validation data

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)



####################################### Creating THE MODEL:
model = Sequential(name="my_CNN_MNIST")

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))



##### optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
epochs = 30
batch_size = 86


###### DATA AUGMENTATION
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)


s=X_train.shape[0] // batch_size
# Fit the model
#print("s=",s)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

results = model.predict(test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
#print("1",results)
results = pd.Series(results,name="Label")
#print("2",results)

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("/home/said/MNIST/cnn_mnist_datagen.csv",index=False)

