"""
"CIFARNET" CNN model architecutre, as described in the paper. 
As our goal is not to tune the model architecture, but experiment with 
optimization method performances, I define the architchture here, and will
not change it elsewhere. 
"""
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam


if K.image_data_format() == "channels_first":
    input_shape = (3, 32, 32)
else:
    input_shape = (32,32,3)

def get_cifar10_cnn(lr=0.01, beta_2 = .99, amsgrad = False, decay = .14):
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