"""
Functions for constructing and compiling models for experiments
""" 

import numpy as np

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy


def get_cifar10_cnn(lr=0.01, beta_2 = .99, amsgrad = False):
    # get the channel config
    if K.image_data_format() == "channels_first":
        input_shape = (3, 32, 32)
    else:
        input_shape = (32,32,3)
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
    optimizer = Adam(lr=lr, beta_2 = beta_2, amsgrad = amsgrad)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    return model


def get_ffnn(lr=0.01, beta_2 = .99, amsgrad = False):
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=784, activation = "relu"))
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    optimizer = Adam(lr=lr, beta_2 = beta_2, amsgrad = amsgrad)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    return model

def get_logreg(lr=0.01, beta_2 = .99, amsgrad = False):
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=784, activation='softmax'))
    # Compile model
    optimizer = Adam(lr=lr, beta_2 = beta_2, amsgrad = amsgrad)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    return model