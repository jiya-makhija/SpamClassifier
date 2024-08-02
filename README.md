# Interactive Spam Classification using TensorFlow

This project is designed to classify SMS messages as spam or ham (not spam) using a neural network model built with TensorFlow and Keras.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Processing](#data-processing)
4. [Model Architecture](#model-architecture)
5. [Usage](#usage)
6. [Prerequisites](#prerequisites)
7. [Features](#features)
8. [Results](#results)
9. [Contact](#contact)

## Project Overview

The aim of this project is to develop a machine learning model that can accurately identify spam messages in a dataset. The model is trained on a collection of SMS messages labeled as either spam or ham. To enhance model accuracy and reliability, the dataset is preprocessed to remove noise and transformed into a suitable format for training using TF-IDF vectorization.

## Dataset

The dataset `spam.csv` used for this project contains SMS messages labeled as spam or ham. It is structured as follows:

- **label**: The class label (ham = 0, spam = 1).
- **message**: The raw SMS message text.

## Data Processing

The dataset undergoes several preprocessing steps to prepare it for model training:

1. **Data Cleaning**: 
   - Convert text to lowercase for uniformity.
   - Tokenize the text into individual words.
   - Remove punctuation and special characters to reduce noise.

2. **Feature Extraction**: 
   - Apply TF-IDF Vectorization to convert the cleaned text into numerical features. This captures the importance of each word relative to the entire dataset.

## Model Architecture

The model, implemented in `spam_classification.ipynb`, is a neural network built using the Keras Sequential API. It consists of the following layers:

- **Input Layer**: Specifies the input shape for the model.
- **Dense Layer 1**: A fully connected layer with ReLU activation.
- **Dense Layer 2**: A second fully connected layer with ReLU activation.
- **Output Layer**: A single neuron with sigmoid activation for binary classification.

## Usage

### Interactive Spam Classification

This project includes an interactive feature that allows you to test the model with your own SMS messages. Follow these steps to use this feature:

1. **Run the Jupyter Notebook**: Open `spam_classification.ipynb` in Jupyter Notebook and navigate to the "User Input Classification" section.
2. **Enter a Message**: Type an SMS message into the input prompt.
3. **View the Prediction**: The notebook will output whether the message is classified as spam or ham.

## Prerequisites

Ensure you have the following installed before running the project:

- Python 3.6 or later
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn

## Features
- Spam Classification Model: Uses a neural network to classify SMS messages into spam or ham categories.
- Interactive User Input: Allows users to input custom SMS messages and receive immediate predictions on their classification.
- Visualization: Provides a confusion matrix heatmap to visually evaluate model performance, highlighting the number of true positives, false positives, true negatives, and false negatives.

## Results
The model achieved high training accuracy (98%) and validation accuracy (96%), indicating effective learning and good generalization. The training and validation loss curves are closely aligned, suggesting minimal overfitting. 

## Contact
  For any questions, feedback, or collaboration opportunities, feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/jiyamakhija/)
