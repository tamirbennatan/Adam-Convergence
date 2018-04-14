import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

import pickle

print("Tuning CNN Neural Network Models on CIFAR-10")


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

X_train = np.load("../data/CIFAR/X_train.npy")
X_test = np.load("../data/CIFAR/X_test.npy")
y_train = np.load("../data/CIFAR/y_train.npy")
y_test = np.load("../data/CIFAR/y_test.npy")

"""
Reshape the data to work with CNN
"""
trainlength, testlength = X_train.shape[0], X_test.shape[0]
# Reshape the X's, according to our channel setting. 
if K.image_data_format() == "channels_last":
    X_train = X_train.reshape(trainlength, 3, 32, 32).transpose(0,2,3,1)
    X_test = X_test.reshape((testlength, 3, 32, 32)).transpose(0,2,3,1)
else:
    X_train = X_train.reshape(trainlength, 3, 32, 32)
    X_test = X_test.reshape((testlength, 3, 32, 32))


"""
Setup gridsearches
"""
"""
Make a hyperparemter grid to search through
"""
beta2_range = [.99, .999]
alpha_range = [.00001, .0001, .001]
param_grid = dict(lr=alpha_range, beta_2=beta2_range)

# """
# Gridsearch through learning rate and beta_2 combinations, using the Adam optimizer
# """
# adam_model = KerasClassifier(build_fn=get_cifar10_cnn, epochs=15, batch_size=128, verbose=1)
# adam_grid = GridSearchCV(estimator=adam_model, param_grid=param_grid, n_jobs=1, verbose=1, cv = 2)
# adam_grid_result = adam_grid.fit(X_train, y_train)

# """
# Print the best parameters, and pickle the best model
# """
# print("Best parameters for CNN with Adam:")
# print(adam_grid_result.best_params_)

# with open("gridsearch_params/cifar_adam.pkl", "wb") as handle:
#     pickle.dump(adam_grid_result.best_params_, handle, protocol = 3)

"""
Gridsearch through learning rate and beta_2 combinations, using the AMSGrad optimizer
"""
# Same grid, but now using AMS optimizer
param_grid_ams = dict(lr=alpha_range, beta_2=beta2_range, amsgrad = [True])

ams_model = KerasClassifier(build_fn=get_cifar10_cnn, epochs=15, batch_size=128, verbose=1)
ams_grid = GridSearchCV(estimator=ams_model, param_grid=param_grid_ams, n_jobs=1, verbose = 1, cv = 2)
ams_grid_result = ams_grid.fit(X_train, y_train)

"""
Print the best parameters, and pickle the best model
"""
print("Best parameters for CNN with AMSGrad Optimizer:")
print(ams_grid_result.best_params_)

with open("gridsearch_params/cifar_ams.pkl", "wb") as handle:
    pickle.dump(ams_grid_result.best_params_, handle,  protocol = 3)

print("Done tuning CIFARNET ")
