import numpy as np
import streamlit as st

from utils import load_data, load_keras_model, pred_sample

LABELS = {0: "Upper part", 1: "Bottom part", 2: "One piece", 3: "Footwear", 4: "Bags"}

st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
)

st.title("Fashion MNIST Classifier")
st.write("This is a web app that uses a neural network to classify images from the Fashion MNIST dataset.")
st.write("Select the index of an image from the test set and click the button to see the prediction.")

# Load data
_, _, test_x, test_y = load_data()

# Load keras model
model = load_keras_model()

sample_id = st.number_input('Select an index of the test set to process:', 
                           min_value=0, max_value=10000, value=0, step=1, format="%d")

st.image(test_x[sample_id], caption=f"Sample {sample_id} from the test set", )

process_button = st.button("Predict image class")

if process_button:
    with st.spinner('Wait for it...'):
        y_pred = pred_sample(model, test_x[sample_id])

    true_label = LABELS[np.argmax(test_y[sample_id])]
    pred_label = LABELS[np.argmax(y_pred)]

    if true_label == pred_label:
        st.success(f"True label: {true_label:>10}\n\n" \
                f"Predicted label: {pred_label:>5}")
    else:
        st.error(f"True label: {true_label:>10}\n\n" \
                f"Predicted label: {pred_label:>5}")
