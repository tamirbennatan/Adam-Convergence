"""
A script for loading the MNIST dataset
And saivng it as numpy binary files
"""
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import numpy as np

# load mnist
mnist = fetch_mldata('MNIST original')
# Extract the data and the labels
X = mnist.data
y = mnist.target

# Make pre-defined train/test split. Test size of 10,000 images.
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=10000, random_state=551)

""" Write all the files. """
def write_np_file(obj, file_name, mode = "wb"):
	with open(file_name, mode) as handle:
		np.save(handle, obj)

write_np_file(X, "X.npy")
write_np_file(y, "y.npy")
write_np_file(X_train, "X_train.npy")
write_np_file(X_test, "X_test.npy")
write_np_file(y_train, "y_train.npy")
write_np_file(y_test, "y_test.npy")