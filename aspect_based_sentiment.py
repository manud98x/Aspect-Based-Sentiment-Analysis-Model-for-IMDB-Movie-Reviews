# Setting  environment variables
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


# List of libraries to check
libraries = ['numpy', 'pandas', 'nltk', 'tensorflow', 'keras']

# Checking if libraries are installed and installing if they're not
for lib in libraries:
    try:
        __import__(lib)
    except ImportError:
        print(f"\n{lib} is not installed. Installing now...")
        os.system(f'pip install {lib}')

# Check if NLTK data resources are installed
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except (LookupError, FileNotFoundError):
    print("\nDownloading NLTK data resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


# Import necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
import tensorflow as tf
from keras.api.datasets import imdb
from keras.api.preprocessing.sequence import pad_sequences
from keras.api.models import Sequential
from keras.api.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Input, Bidirectional,BatchNormalization, Dropout
from keras.api.regularizers import l2
from keras.api.callbacks import EarlyStopping
from keras.api.initializers import HeUniform
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


print("\n")
print("╔═════════════════════════════════════════════════╗")
print("║ Assignment 02 - Aspect Based Sentiment Analysis ║")
print("║                  Group - D                      ║")
print("╚═════════════════════════════════════════════════╝")
print("\n")

# Define preprocessing function  
def preprocess_text_lemmatization(text):

    # Tokenize the text
    tokens = word_tokenize(text)
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    # Lemmatize words, remove stopwords and non-alphanumeric words
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalnum()]
    return ' '.join(tokens)

# Load and preprocess data using lemmatization
def load_and_preprocess_data(words=10000,max_len=500):
    
    print("Fetching Data from the Repository....")

    # Load IMDB dataset
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=words)

    # Get word index
    word_index = imdb.get_word_index()

    # Reverse word index
    reverse_word_index = {value: key for key, value in word_index.items()}

    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=words)

    # Decode train reviews
    decoded_train_reviews = [' '.join([reverse_word_index.get(i - 3, '?') for i in sequence]) for sequence in train_data]

    # Decode test reviews
    decoded_test_reviews = [' '.join([reverse_word_index.get(i - 3, '?') for i in sequence]) for sequence in test_data]

    print("Pre-Processing Train Reviews....")

    # Preprocess train reviews
    preprocessed_train_reviews = [preprocess_text_lemmatization(review) for review in decoded_train_reviews]
    print("Pre-Processing Test Reviews....")
    
    # Preprocess test reviews
    preprocessed_test_reviews = [preprocess_text_lemmatization(review) for review in decoded_test_reviews]

    # Fit tokenizer on preprocessed train reviews
    tokenizer.fit_on_texts(preprocessed_train_reviews)

    # Convert text to sequences
    sequences_train = tokenizer.texts_to_sequences(preprocessed_train_reviews)
    sequences_test = tokenizer.texts_to_sequences(preprocessed_test_reviews)

    # Get maximum sequence length
    maxlen = max(len(seq) for seq in sequences_train)
    maxlen = min(maxlen, max_len)

    # Pad sequences
    train_reviews = pad_sequences(sequences_train, maxlen=maxlen)
    test_reviews = pad_sequences(sequences_test, maxlen=maxlen)

    print("Pre-Processing Completed!")

    return train_reviews, test_reviews, train_labels, test_labels

# Set constants
MAX_FEATURES = 1000
INPUT_LENGTH = 100   

# Load and preprocess data
train_reviews, test_reviews, train_labels, test_labels = load_and_preprocess_data(words=MAX_FEATURES,max_len=INPUT_LENGTH)

# Initialize model
cnn_bi_lstm_model = Sequential()

# Add input layer
cnn_bi_lstm_model.add(Input(shape=(INPUT_LENGTH,))) 

# Add embedding layer
cnn_bi_lstm_model.add(Embedding(MAX_FEATURES, 128, ))  

# Add convolutional layer
cnn_bi_lstm_model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer=HeUniform()))

# Add batch normalization layer
cnn_bi_lstm_model.add(BatchNormalization()) 

# Add max pooling layer
cnn_bi_lstm_model.add(MaxPooling1D(pool_size=2))

# Add bidirectional LSTM layer
cnn_bi_lstm_model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5)))

# Add another batch normalization layer
cnn_bi_lstm_model.add(BatchNormalization()) 

# Add dense layer
cnn_bi_lstm_model.add(Dense(64, activation='relu'))

# Add output layer
cnn_bi_lstm_model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))

# Compile model
cnn_bi_lstm_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print model summary
cnn_bi_lstm_model.summary()

# Use early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

print("\nTraining the CNN-BiLSTM model...\n")

# Train model
cnn_bi_lstm_model.fit(train_reviews, train_labels,
                    validation_split=0.1,
                    epochs=3,
                    batch_size=512,
                    callbacks=[early_stopping])

print("\nModel Training Done! \n")

print("Evaluating Model....\n")

# Model to evaluate
models = [cnn_bi_lstm_model]

# Model names
model_names = ['CNN-BiLSTM']

# Initialize table
table = """
┌───────────────────────┬─────────────────┬─────────────┐
│         Model         │  Test Accuracy  │  Test Loss  │
├───────────────────────┼─────────────────┼─────────────┤
"""

# Evaluate each model
for i, model in enumerate(models):
    loss, accuracy = model.evaluate(test_reviews, test_labels, verbose=0)
    # Add model results to table
    table += "│ {:<21} │ {:<15.2f} │ {:<11.2f} │\n".format(model_names[i], accuracy, loss)

# Close table
table += "└───────────────────────┴─────────────────┴─────────────┘"

# Print table
print(table)

# Print note
print("""\n**Note**: 
      For the purpose of optimizing execution time,
      certain parameters within our model have been set to lower values.
      This approach facilitates quicker computational results.
      However, it's important to note that this may impact the performance of the model

      Changes Made:
      MAX_FEATURES: Set to 1000 (10000 in the Original)
      INPUT_LENGTH: Set to 100 (500 in the Original)
      Epoch: Limited to 3  (10 in the Original)
      Batch_Size: Set to 512 (64 in the Original)
      """)

print()
