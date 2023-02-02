import pickle
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
