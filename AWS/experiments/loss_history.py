from keras.callbacks import Callback
import pandas as pd
import pdb

"""
A custom Keras callback, used to store metrics on a per-batch basis, as
opposed to a on a per-epoch basis - which the default history callback uses. 

Metrics:
    - training loss
    - test loss
    - training accuracy
    - test accuracy

Note - computing the validation loss/accuracy is slow. It should not be done at 
the end of every epoch. Therefore, the constructor of this callback requires an
<every> parameter, which specifies the interval of batches with which to calculate 
the validation accuracy/loss. 

How to set <every>? Well, the point of calculating loss on a per-batch basis is so that
you can get a more fine tuned view of how the model is converging. So if you want to tune
for 20 epochs, and have a line chart in mind, perhaps it would be nice to have 200 points to 
plot (10 per epoch), so you would set every to (number of data points)/(batch size * 10)
"""

class LossHistory(Callback):

    def __init__(self, X_test, y_test, every):
        self.X_test = X_test
        self.y_test = y_test
        self.every = every
        self.batches_seen = 0

    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []

    def on_batch_end(self, batch, logs={}):
        if self.batches_seen % self.every == 0:
            self.train_loss.append(logs.get('loss'))
            self.train_acc.append(logs.get('acc'))
            # isolate the test data
            X_test, y_test = self.X_test, self.y_test
            val_loss, val_acc = self.model.evaluate(X_test, y_test, verbose=0)
            self.valid_loss.append(val_loss)
            self.valid_acc.append(val_acc)
        self.batches_seen += 1

    def on_train_end(self, logs = {}):
        # return the entire training log as a pandas dataframe
        self.log = pd.DataFrame({
            "train_loss" : self.train_loss, 
            "train_acc"  : self.train_acc,
            "valid_loss" : self.valid_loss, 
            "valid_acc"  : self.valid_acc})