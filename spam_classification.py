import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk import word_tokenize
from sklearn.preprocessing import LabelEncoder  # converts categorical labels into numerical values.
from sklearn.model_selection import train_test_split

spam_data = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1], names=['label', 'message'], skiprows=1)
# converting label to 0 for ham and 1 for spam
label_encoder = LabelEncoder()
spam_data['label'] = label_encoder.fit_transform(spam_data['label'])

# Data Processing
# Removing Punctuations & Special Characters
nltk.download('punkt')


def process_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    cleaned_text = ' '.join(tokens)
    return cleaned_text


spam_data['cleaned_message'] = spam_data['message'].apply(process_text)
spam_data.drop(columns=['message'], inplace=True)

# Splitting the data into training (60%), validation (20%), and test (20%)
train_data, temp_data, train_labels, temp_labels = train_test_split(
    spam_data['cleaned_message'],
    spam_data['label'],
    test_size=0.4,
    random_state=42,
)

val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data,
    temp_labels,
    test_size=0.5,
    random_state=42,
)
