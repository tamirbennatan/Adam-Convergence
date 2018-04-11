"""
A script for loading the CIFAR-10 dataset and 
writing to numpy binary files. 
Train/test splits are already made, based on author's pre-defined splits. 

In this script I refer to the data repository that sits in my local Downloads folder. 
To download this file yourself, see: http://www.cs.toronto.edu/~kriz/cifar.html

This is a python-2.* script.
"""

import numpy as np 
import cPickle
# path to the data folder.
DATA_DIR = "/Users/timibennatan/Downloads/cifar-10-batches-py/"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def write_np_file(obj, file_name, mode = "wb"):
    with open(file_name, mode) as handle:
        np.save(handle, obj)

# accumuate the training data/labels from the first 5 batches
data, labels = [], []
for i in range(1,6):
    filename = DATA_DIR + ("data_batch_%d" %(i))
    # unpickle the data
    unpickled = unpickle(filename)
    # add extracted data/labels
    data.append(unpickled["data"])
    labels.append(unpickled["labels"])

# Consolidate one training set from all the batches
X_train = np.vstack(data)
y_train = np.array([l for batch in labels for l in batch])

# Now load the test data
test_file = DATA_DIR + "test_batch"
unpickled = unpickle(filename)
X_test = (unpickled["data"])
y_test = unpickled["labels"]


"""
Save everything as numpy binary files
"""
write_np_file(X_train, "X_train.npy")
write_np_file(X_test, "X_test.npy")
write_np_file(y_train, "y_train.npy")
write_np_file(y_test, "y_test.npy")