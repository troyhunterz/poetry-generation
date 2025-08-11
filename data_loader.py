import string
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

filenames = ['dataset/adele.txt', 'dataset/lady-gaga.txt',
             'dataset/kanye-west.txt', 'dataset/eminem.txt']


with open('dataset/MERGED.txt', 'w', encoding='utf-8') as outfile:
    for fname in filenames:
        with open(fname, 'r', encoding='utf-8') as infile:
            outfile.write(infile.read() + '\n')


with open('dataset/MERGED.txt', 'r', encoding='utf-8') as file:
    data = file.read().splitlines()


# Building LSTM Model
token = Tokenizer()
token.fit_on_texts(data)
encoded_text = token.texts_to_sequences(data)
vocab_size = len(token.word_counts) + 1


# Prepare Training Data
datalist = []
for d in encoded_text:
    if len(d) > 1:
        for i in range(2, len(d)):
            datalist.append(d[:i])


# Padding
max_length = 50
sequences = pad_sequences(datalist, maxlen=max_length, padding='pre')

X = sequences[:, :-1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)

seq_length = X.shape[1]

__all__ = ['X', 'y', 'token', 'seq_length', 'vocab_size']
