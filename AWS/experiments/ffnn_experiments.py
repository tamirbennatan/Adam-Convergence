"""
Run the Feedforwarnd neural network experiments. 
Run <epochs> epochs, to be passed as a command line argument.
Repeat for <runs> runs, with different random initialializations each time. 
Store training and test losses, as well as accuracies, on a per-batch basis.
"""

import numpy as np
import pandas as pd

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import np_utils

import argparse
import yaml
import datetime

# custom functions/classes
from models import get_ffnn
from loss_history import LossHistory
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
# isolate hyperparameters regarding FFNN
ffnn_params = hyperparam_config["FFNN"]

"""
Keep track of a log of all the runs, and the relevent metrics:
    - training loss
    - test loss
    - training accuracy
    - test accuracy
"""
experiment_log = pd.DataFrame({
    "run" : [], 
    "batch" : [],
    "optimizer" : [],
    "train_loss" : [],
    "valid_loss" : [],
    "train_acc" : [],
    "valid_acc" : []}
    )

"""
Load the MNIST data.
"""
X_train = np.load("../data/MNIST/X_train.npy") 
X_test = np.load("../data/MNIST/X_test.npy") 
y_train = np.load("../data/MNIST/y_train.npy") 
y_test = np.load("../data/MNIST/y_test.npy") 

# convert the labels to a one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

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
        alpha = ffnn_params[optimizer]["alpha"]
        beta_2 = ffnn_params[optimizer]["beta"]
        
        # Get a  brand new model for each run - we need new random initializations
        model = get_ffnn(lr = alpha, beta_2 = beta_2, amsgrad = amsgrad)
        # Construct a new History monitor, so that we can keep track of the losses.
        # record the validation loss/accuracy every 100 batches.
        run_history = LossHistory(X_test, y_test, every = 100)
        print()
        print("Training model with <%s> optimizer, run <%d>." %(optimizer, run))
        # train that ish for <epochs> epochs.
        _ = model.fit(X_train, y_train, batch_size=batch, epochs=epochs, verbose=1, 
            callbacks = [run_history])
        print()
        # pdb.set_trace()

        """
        Extract the log, and add it to the experiment_history
        """
        run_log = run_history.log 
        # Add columns for the run and the optimizer used, as well as the batch number
        run_log['run'] = run
        run_log['optimizer'] = optimizer
        run_log['batch'] = range(run_log.shape[0])
        # append to the experiment hisory
        experiment_log = pd.concat((experiment_log,run_log))

"""
At the end of the experiment, write the experiment history to disk.
"""
now = datetime.datetime.now()
filename = "experiment_log/ffnn_%d-%d_%d-%d.csv" %(now.month, now.day, now.hour, now.minute)
experiment_log.to_csv(filename)