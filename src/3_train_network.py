import pickle

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

def load_data():
    """
    Load data from pickle file
    """
    with open('../data/processed/fashion_mnist_k5.pkl', 'rb') as f:
        (train_x, train_y), (test_x, test_y) = pickle.load(f)
        return train_x, train_y, test_x, test_y

def cast_and_normalize_images(train, test):
    """
    Convert from integers to floats and normalize to range 0-1
    """
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    
    return train_norm, test_norm


def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # Prepare data
    train_x, train_y, test_x, test_y = load_data()
    train_x, test_x = cast_and_normalize_images(train_x, test_x)
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, 
        train_y,
        test_size=0.2, 
        shuffle=True,
        random_state=42,
    )

    # Train model
    model = get_model()
    model.fit(train_x, train_y, epochs=2, batch_size=16, validation_data=(val_x, val_y), verbose=1)

    # Evaluate model
    y_pred = model.predict(test_x)
    print(classification_report(np.argmax(test_y, axis=1), np.argmax(y_pred, axis=1)))
    print(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(y_pred, axis=1)))

    # Save keras model to disk
    model.save('../models/keras_model.h5')
    print("Model saved to disk")
