#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

# Example news articles (replace with your dataset)
news_data = [
    "This is a sample news article about a topic.",
    "Another news article with different content.",
    "Fake news can mislead people easily.",
    "Authentic news sources are essential for accurate information.",
]

# Initialize a tokenizer with one-hot encoding
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

# Fit the tokenizer on your dataset
tokenizer.fit_on_texts(news_data)

# Convert text data to one-hot vectors
one_hot_results = tokenizer.texts_to_matrix(news_data, mode="binary")

index_to_display = 0
one_hot_example = one_hot_results[index_to_display]

print(f"One-Hot Encoded Data for Example {index_to_display + 1}:\n")
print(one_hot_example)


# In[5]:


# If you want to display the one-hot encoded data as a heatmap:
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.imshow([one_hot_example], cmap='binary', aspect='auto')
plt.title('One-Hot Encoded Data for Example')
plt.xlabel('Word Index in Sequence')
plt.yticks([])
plt.show()


