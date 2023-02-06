import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

from utils import load_data


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
    """
    Define the CNN model
    """
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


def plot_confusion_matrix(test_y, y_pred):
    """
    Plot confusion matrix

    Parameters
    -----------
    test_y:
        True labels in categorical format
    y_pred:
        Predicted labels in categorical format
    """
    ConfusionMatrixDisplay.from_predictions(np.argmax(test_y, axis=1), np.argmax(y_pred, axis=1), 
                                            normalize="true", 
                                            include_values=True,
                                            cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")

    plt.savefig('../plots/confusion_matrix.png', dpi=300, bbox_inches='tight')


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

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(train_x, train_y, 
              epochs=16, batch_size=16, 
              validation_data=(val_x, val_y), 
            #   class_weight={0: 0.25, 1: 1, 2: 1, 3: 0.333, 4: 1},
              callbacks=[reduce_lr, early_stop], 
              verbose=1)

    # Evaluate model
    y_pred = model.predict(test_x)
    print(classification_report(np.argmax(test_y, axis=1), np.argmax(y_pred, axis=1)))
    print(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(y_pred, axis=1)))
    plot_confusion_matrix(test_y, y_pred)

    # Save keras model to disk
    model.save('../models/keras_model.h5')
    print("\nModel saved to disk!")
