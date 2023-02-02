# Fashion-MNIST Classifier

Project to classify images from the Fashion-MNIST dataset using Keras. Instead of using the original dataset, this project merges some of the classes to make it a 5-class problem.

# Project Structure

The project is structured as follows:

```bash
.
├── data
│   ├── original        # Original dataset
│   └── processed       # 5 class dataset
├── Dockerfile
├── models              # Keras models
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

# Setup

I recommend using VSCode with the [Remote Containers](https://code.visualstudio.com/docs/remote/containers) extension. This will allow you to run the code in a docker container with all the dependencies installed using the Dockerfile.

The Dockerfile is based on the official tensorflow docker image with `python 3.8` and takes care of installing the dependencies.

If you prefer to build your own environment, you can use the requirements.txt file.

# Streamlit App

I've built a simple streamlit app to test the model. You can run it with the following command:

```bash
cd src
streamlit run 5_predictor_streamlit.py
```

## Screenshot

<img src="streamlit_screenshot.png">


# Results

The model was trained for 6 epochs with a batch size of 16. The results are as follows:

```bash
              precision    recall  f1-score   support

           0       0.97      0.98      0.98      4000
           1       0.99      0.98      0.98      1000
           2       0.91      0.90      0.90      1000
           3       1.00      1.00      1.00      3000
           4       0.99      0.97      0.98      1000

    accuracy                           0.98     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.98      0.98      0.98     10000



[[3914    2   72    1   11]
 [   7  980   13    0    0]
 [  89    7  902    1    1]
 [   1    0    0 2999    0]
 [  16    1    7    7  969]]
 ```

 As we can see, the model performs very well with an accuracy of 98%. All the classes have a F1 score of 0.98 or higher, except for the "One Piece" class, which has a F1 score of 0.90. With the confusion matrix we can see that the model is having some trouble between the "One Piece" and "Upper Part" classes.
 