import numpy as np

from utils import load_data, load_keras_model, pred_sample

LABELS = {0: "Upper part", 1: "Bottom part", 2: "One piece", 3: "Footwear", 4: "Bags"}


if __name__ == '__main__':
    # Load data
    train_x, train_y, test_x, test_y = load_data()

    # Load keras model
    model = load_keras_model()

    while True:

        # Enter id of sample to predict with keyboard
        try:
            sample_id = int(input("Enter sample id to predict: "))
        except ValueError:
            print("Invalid id. The id must be a whole number\n\n")
            continue

        if not isinstance(sample_id, int):
            print("Invalid id. The id must be a whole number\n\n")
            continue
        
        if sample_id > 10000 or sample_id < 0:
            print("Invalid id. The id must be between 0 and 10000\n\n")
            continue

        # Predict
        y_pred = pred_sample(model, test_x[sample_id])

        true_label = LABELS[np.argmax(test_y[sample_id])]
        pred_label = LABELS[np.argmax(y_pred)]

        print("--------------------------------")
        print(f"True label: {true_label:>18}")
        print(f"Predicted label: {pred_label:>13}")
        print("--------------------------------\n\n")
