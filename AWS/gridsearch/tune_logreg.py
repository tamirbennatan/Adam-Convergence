import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
import pickle

print("Tuning Logistic Regressions.")

# A function that, when passed with hyperparameter options, returns a compiled model
# Note that if `amsgrad = True`, the method in the paper is used.
def create_model(lr=0.01, beta_2 = .99, amsgrad = False):
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=784, activation='softmax'))

    
    # Compile model
    optimizer = Adam(lr=lr, beta_2 = beta_2, amsgrad = amsgrad, decay = .14)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

X_train = np.load("../data/MNIST/X_train.npy") 
X_test = np.load("../data/MNIST/X_test.npy") 
y_train = np.load("../data/MNIST/y_train.npy") 
y_test = np.load("../data/MNIST/y_test.npy") 

"""
Make a hyperparemter grid to search through
"""
beta2_range = np.append(np.arange(.990, .999, .0025), .999)
alpha_range = [.0001*10**i for i in range(5)]
param_grid = dict(lr=alpha_range, beta_2=beta2_range)

"""
Gridsearch through learning rate and beta_2 combinations, using the Adam optimizer
"""
adam_model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=128, verbose=1)
adam_grid = GridSearchCV(estimator=adam_model, param_grid=param_grid, n_jobs=1, verbose=1)
adam_grid_result = adam_grid.fit(X_train, y_train)

"""
Print the best parameters, and pickle the best model
"""
print("Best parameters for Logistic regression with Adam:")
print(adam_grid_result.best_params_)

with open("gridsearch_params/logreg_adam.pkl", "wb") as handle:
    pickle.dump(adam_grid_result.best_params_, handle, protocol = 3)


"""
Gridsearch through learning rate and beta_2 combinations, using the AMSGrad optimizer
"""
# Same grid, but now using AMS optimizer
param_grid_ams = dict(lr=alpha_range, beta_2=beta2_range, amsgrad = [True])

ams_model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=128, verbose=1)
ams_grid = GridSearchCV(estimator=ams_model, param_grid=param_grid_ams, n_jobs=1, verbose = 1)
ams_grid_result = ams_grid.fit(X_train, y_train)

"""
Print the best parameters, and pickle the best model
"""
print("Best parameters for Logistic regression with AMSGrad Optimizer:")
print(ams_grid_result.best_params_)

with open("gridsearch_params/logreg_ams.pkl", "wb") as handle:
    pickle.dump(ams_grid_result.best_params_, handle, protocol = 3)

print("Done tuning Logistic Regressions. ")