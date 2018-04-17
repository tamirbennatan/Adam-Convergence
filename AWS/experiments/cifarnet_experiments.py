"""
Run the Logistic Regression experiments. 
Run <epochs> epochs, to be passed as a command line argument.
Repeat for <runs> runs, with different random initialializations each time. 
Store training and test losses, as well as accuracies, on a per-batch basis.
"""

import numpy as np
import pandas as pd

import keras.backend as K
from keras.optimizers import Adam
from keras.utils import np_utils

import argparse
import yaml
import datetime

# custom functions/classes
from models import get_cifar10_cnn
import pdb


# get command line arguments for the number of epochs, and number of runs
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", dest = "epochs", type = int, default = 20, 
                    help = "How many epochs to train per round.")
parser.add_argument("--runs", dest = "runs", type = int, default = 1, 
                    help = "How many runs would you like to run per classifier?")
parser.add_argument("--batch", dest = "batch", type = int, default = 128, 
                    help = "Batch size for mini-batch updats")
args = parser.parse_args()
epochs = args.epochs
runs = args.runs
batch = args.batch

"""
load the tuned hyperparameters as a dictionary
"""
with open("param_config.yml", "r") as handle:
    hyperparam_config = yaml.load(handle)
handle.close()
# isolate hyperparameters regarding Logistic Regression
cinfarnet_params = hyperparam_config["CifarNet"]

"""
Keep track of a log of all the runs, and the relevent metrics:
    - training loss
    - test loss
    - training accuracy
    - test accuracy
"""
experiment_log = pd.DataFrame({
    "run" : [], 
    "epoch" : [],
    "optimizer" : [],
    "train_loss" : [],
    "valid_loss" : [],
    "train_acc" : [],
    "valid_acc" : []}
    )

"""
Load the MNIST data.
"""
X_train = np.load("../data/CIFAR/X_train.npy") 
X_test = np.load("../data/CIFAR/X_test.npy") 
y_train = np.load("../data/CIFAR/y_train.npy") 
y_test = np.load("../data/CIFAR/y_test.npy") 

# convert the labels to a one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


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
For each of the possible optimizers,
train for <epochs> epochs for each of the <runs> runs.
"""
for optimizer in ["Adam", "AMSGrad"]:
    # train for <runs> runs for each optimizer
    for run in range(runs):
        # are we using Adam, or AMSGrad? 
        amsgrad = optimizer == "AMSGrad"
        # isolate the other hyperaparmeters
        alpha = cinfarnet_params[optimizer]["alpha"]
        beta_2 = cinfarnet_params[optimizer]["beta"]
        
        # Get a  brand new model for each run - we need new random initializations
        model = get_cifar10_cnn(lr = alpha, beta_2 = beta_2, amsgrad = amsgrad)
        # Construct a new History monitor, so that we can keep track of the losses.
        # record the validation loss/accuracy every 100 batches.
        print()
        print("Training model with <%s> optimizer, run <%d>, alpha = <%f>, beta2 = <%f>" %(optimizer, run, alpha, beta_2))
        # train that ish for <epochs> epochs.
        run_history = model.fit(X_train, y_train, batch_size=batch, epochs=epochs, verbose=1, validation_data = (X_test, y_test))
        print()

        """
        Extract the log, and add it to the experiment_history
        """
        run_log = pd.DataFrame({"train_loss": run_history.history['loss'],
            "train_acc"  : run_history.history['categorical_accuracy'], 
            "valid_loss" : run_history.history['val_loss'], 
            "valid_acc"  : run_history.history['val_categorical_accuracy']})
        # Add columns for the run and the optimizer used
        run_log['run'] = run
        run_log['optimizer'] = optimizer
        run_log['epoch'] = range(run_log.shape[0])
        # append to the experiment hisory
        experiment_log = pd.concat((experiment_log,run_log))

"""
At the end of the experiment, write the experiment history to disk.
"""
now = datetime.datetime.now()
filename = "experiment_log/cifarnet_%d-%d_%d-%d.csv" %(now.month, now.day, now.hour, now.minute)
experiment_log.to_csv(filename)