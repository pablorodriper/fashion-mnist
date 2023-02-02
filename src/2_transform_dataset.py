import gzip
import os
import pickle

import numpy as np
from tensorflow.keras.utils import to_categorical


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def change_labels(labels):
    """Change dataset from 10 classes to 5 classes"""

    labels = np.where(labels == 0, 0, labels)
    labels = np.where(labels == 1, 1, labels)
    labels = np.where(labels == 2, 0, labels)
    labels = np.where(labels == 3, 2, labels)
    labels = np.where(labels == 4, 0, labels)
    labels = np.where(labels == 5, 3, labels)
    labels = np.where(labels == 6, 0, labels)
    labels = np.where(labels == 7, 3, labels)
    labels = np.where(labels == 8, 4, labels)
    labels = np.where(labels == 9, 3, labels)

    return labels
    

# Load data
X_train, y_train = load_mnist('../data/original', kind='train')
X_test, y_test = load_mnist('../data/original', kind='t10k')

# Reshape data
trainX = X_train.reshape((X_train.shape[0], 28, 28, 1))
testX = X_test.reshape((X_test.shape[0], 28, 28, 1))

# Change labels
y_train = change_labels(y_train)
y_test = change_labels(y_test)

# Reshape labels
trainY = to_categorical(y_train)
testY = to_categorical(y_test)

with open("../data/processed/fashion_mnist_k5.pkl", "wb") as f:
    pickle.dump(((trainX, trainY), (testX, testY)), f)