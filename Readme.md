# Fashion-MNIST Classifier


# Structure

```bash
.
├── data
│   ├── original
│   │   ├── t10k-images-idx3-ubyte.gz
│   │   ├── t10k-labels-idx1-ubyte.gz
│   │   ├── train-images-idx3-ubyte.gz
│   │   └── train-labels-idx1-ubyte.gz
│   └── processed
│       └── fashion_mnist_k5.pkl
├── Dockerfile
├── models
│   └── keras_model.h5
├── notebooks
├── Readme.md
├── requirements.txt
└── src
    ├── 1_download_dataset.py
    ├── 2_transform_dataset.py
    ├── 3_train_network.py
    ├── 4_predictor.py
    ├── 5_predictor_streamlit.py
    └── utils.py
```


# Streamlit App

<img src="streamlit_screenshot.png">

## Run

```bash
cd src
streamlit run 5_predictor_streamlit.py
```

# Results

```bash
              precision    recall  f1-score   support

           0       0.97      0.97      0.97      4000
           1       0.99      0.97      0.98      1000
           2       0.88      0.91      0.89      1000
           3       1.00      1.00      1.00      3000
           4       0.97      0.98      0.97      1000

    accuracy                           0.97     10000
   macro avg       0.96      0.97      0.96     10000
weighted avg       0.97      0.97      0.97     10000


>> Confussion matrix
[[3870    1  101    2   26]
 [  10  971   18    0    1]
 [  83    5  910    1    1]
 [   0    0    0 2999    1]
 [  10    1    6    6  977]]
 ```