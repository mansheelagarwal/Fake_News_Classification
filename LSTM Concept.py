#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Define the maximum number of words (vocabulary size)
vocab_size = len(tokenizer.word_index) + 1  # Ensure you have tokenizer from previous code

# Define the embedding dimension
embedding_dim = 100

# Create an LSTM model
model = Sequential()

# Add an embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))

# Add an LSTM layer
model.add(LSTM(units=64, return_sequences=True))  # Adjust units as needed

# Add an output layer for binary classification
model.add(Dense(units=1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Display the model architecture
model.summary()


# In[7]:


clear

