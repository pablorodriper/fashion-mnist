import pickle

import numpy as np
from tensorflow.keras.models import load_model


def load_data():
    """
    Load data from pickle file
    """
    with open('../data/processed/fashion_mnist_k5.pkl', 'rb') as f:
        (train_x, train_y), (test_x, test_y) = pickle.load(f)
        return train_x, train_y, test_x, test_y


def load_keras_model():
    """
    Load keras model from h5 file
    """
    return load_model('../models/keras_model.h5')


def pred_sample(model, sample):
    """
    Predict sample with keras model. It is necessary to add a batch dimension (1, 28, 28, 1)

    model: keras model
    sample: numpy array of shape (28, 28, 1)
    """
    sample = np.expand_dims(sample, axis=0)
    return model.predict(sample, verbose=0)
