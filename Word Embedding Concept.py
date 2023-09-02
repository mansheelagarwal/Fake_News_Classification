#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example news articles (replace with your dataset)
news_data = [
    "This is a sample news article about a topic.",
    "Another news article with different content.",
    "Fake news can mislead people easily.",
    "Authentic news sources are essential for accurate information.",
]

# Initialize a tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

# Fit the tokenizer on your dataset
tokenizer.fit_on_texts(news_data)

# Convert text data to sequences using word embedding
sequences = tokenizer.texts_to_sequences(news_data)

# Pad sequences to make them uniform in length
padded_sequences = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")

# Define the maximum number of words (vocabulary size)
vocab_size = len(tokenizer.word_index) + 1

# Define the embedding dimension
embedding_dim = 100

# Create an embedding layer
embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

# Apply word embedding to your data (replace input_data with your dataset)
embedded_data = embedding_layer(padded_sequences)

# Now, embedded_data contains word embeddings for your news articles


# In[2]:


import matplotlib.pyplot as plt

# Choose an index to display the embedded data 
index_to_display = 0

# Get the embedded data for the selected index
embedded_example = embedded_data[index_to_display]

# Display the embedded data as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(embedded_example, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Embedded Data for Example')
plt.xlabel('Word Embedding Dimensions')
plt.ylabel('Word Index in Sequence')
plt.show()

