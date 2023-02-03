import gzip
import os
import pickle

import numpy as np
from tensorflow.keras.utils import to_categorical


def load_mnist(path, kind='train'):
    """
    Load MNIST data from `path`

    Code from the Fashion-MNIST repository.
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py

    Parameters
    ----------
    path : str
        Path to the MNIST data directory
    kind : str
        Either 'train' or 't10k' to load the training or test dataset
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def change_labels(labels):
    """
    Change dataset from 10 classes to 5 classes with the following mapping:
        Upper part: T-shirt/top + Pullover + Dress + Shirt
        Bottom part: Trouser
        One piece: Dress
        Footwear: Sandal + Sneaker + Ankle boot
        Bags: Bag

    Parameters
    ----------
    labels : numpy.ndarray
        Labels to be changed. Each label is an integer between 0 and 9.
    
    Returns
    -------
    labels : numpy.ndarray
        Changed labels. Each label is an integer between 0 and 4.
    """

    # labels = np.where(labels == 0, 0, labels)     # Added for readability
    # labels = np.where(labels == 1, 1, labels)     # Added for readability
    labels = np.where(labels == 2, 0, labels)
    labels = np.where(labels == 3, 2, labels)
    labels = np.where(labels == 4, 0, labels)
    labels = np.where(labels == 5, 3, labels)
    labels = np.where(labels == 6, 0, labels)
    labels = np.where(labels == 7, 3, labels)
    labels = np.where(labels == 8, 4, labels)
    labels = np.where(labels == 9, 3, labels)

    return labels
    
if __name__ == "__main__":
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