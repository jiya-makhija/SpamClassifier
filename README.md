# Spam Classification using TensorFlow

This project is designed to classify SMS messages as spam or ham (not spam) using a neural network model built with TensorFlow and Keras.

## Project Overview

The aim of this project is to develop a machine learning model that can accurately identify spam messages in a dataset. The model is trained on a dataset of SMS messages that are labeled as either spam or ham. The dataset has been preprocessed to remove noise and transformed into a suitable format for training.

## Dataset

The dataset (spam.csv) used for this project contains SMS messages that are labeled as spam or ham. It is structured as follows:

- **label**: The class label (ham = 0, spam = 1).
- **message**: The raw SMS message text.
- **cleaned_message**: The preprocessed text after removing punctuation and special characters.

## Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.6 or later
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn

## Model Architecture

The model (in spam_classification.ipynb) is a neural network built using the Keras Sequential API. It consists of the following layers:

- **Input Layer**: Defines the input shape for the model.
- **Dense Layer 1**: A fully connected layer with ReLU activation.
- **Dense Layer 2**: A second fully connected layer with ReLU activation.
- **Output Layer**: A single neuron with sigmoid activation for binary classification.
