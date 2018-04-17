import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

from keras.callbacks import EarlyStopping

import pickle

print("Tuning CNN Neural Network Models on CIFAR-10")


if K.image_data_format() == "channels_first":
    input_shape = (3, 32, 32)
else:
    input_shape = (32,32,3)
def get_cifar10_cnn(lr=0.01, beta_2 = .99, amsgrad = True, decay = .01):
    model = Sequential()
    """
    Two convolutional layers, with Max-pooling in between. 
    64 filters, with a receptive field (kernel) of 6x6. 
    """
    model.add(Conv2D(64, (6,6), padding='same',
                 input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (6,6), padding='same',
                 input_shape=input_shape))
    model.add(Activation('relu'))
    """
    Two dense layers. 
    A Dropout with .5 retention probability is used.
    """
    model.add(Flatten())
    model.add(Dense(384))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    """
    Compile model with optimization hyperparameters
    """
    optimizer = Adam(lr=lr, beta_2 = beta_2, amsgrad = amsgrad, decay = decay)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

X_train = np.load("../data/CIFAR/X_train.npy")
# X_test = np.load("data/CIFAR/X_test.npy")
y_train = np.load("../data/CIFAR/y_train.npy")
# y_test = np.load("data/CIFAR/y_test.npy")
y_train = np_utils.to_categorical(y_train)

"""
Reshape the data to work with CNN
"""
trainlength = X_train.shape[0]
# Reshape the X's, according to our channel setting. 
if K.image_data_format() == "channels_last":
    X_train = X_train.reshape(trainlength, 3, 32, 32).transpose(0,2,3,1)
    # X_test = X_test.reshape((testlength, 3, 32, 32)).transpose(0,2,3,1)
else:
    X_train = X_train.reshape(trainlength, 3, 32, 32)
    # X_test = X_test.reshape((testlength, 3, 32, 32))

callbacks = [
    EarlyStopping(monitor='loss', patience=, verbose=0),
]

"""
Make a hyperparemter grid to search through
"""
beta2_range = [.999, .99]
alpha_range = [ .0001, .001, .00001]

best_beta = None
best_alpha = None
best_acc = -1

for beta in beta2_range:
    for alpha in alpha_range:
        # get a model
        model = get_cifar10_cnn(lr = alpha, beta_2 = beta)
        # train for 25 epochs
        print("TRAINING: alpha = %f, beta2 = %f" %(alpha, beta))
        history = model.fit(X_train, y_train, epochs = 25, callbacks = call)

        
# get the training accuracy
        acc = max(history.history['acc'])
        if acc > best_acc:
            best_acc = acc
            best_beta = beta
            best_alpha = alpha

print("Best alpha " + str(best_alpha))
print("Best beta" + str(best_beta))

best_params = {"alpha": best_alpha, "beta2": best_beta}
with open("cifarnet_ams.pkl", "wb") as handle:
    pickle.dump(best_params, handle)