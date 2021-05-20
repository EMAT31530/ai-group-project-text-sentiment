import sklearn
import csv
import numpy as np
import math
import os
import sys
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, concatenate, Dropout, concatenate, Bidirectional
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split


df = pd.DataFrame()
df = pd.read_csv('/Datasets/cleaned_tweets.csv', encoding = 'utf-8', usecols = ['SpellCheckTweets', 'Sentiment'])
df.head()

df['SpellCheckTweets'] = df['SpellCheckTweets'].astype('str')

X = df['SpellCheckTweets']
y = df['Sentiment']

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=200)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

train_seq_x = sequence.pad_sequences(
    tokenizer.texts_to_sequences(x_train), maxlen=70)
valid_seq_x = sequence.pad_sequences(
    tokenizer.texts_to_sequences(x_test), maxlen=70)

embeddings_index = {}
for i, line in enumerate(open('/Embeddings/Word2Vec_embedding.txt', encoding="utf8")):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

def readfile():
    avgEmbeddedVectors = []
    with open('/Datasets/cleaned_tweets.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            summedVec = [0]*100
            sentenceSplit = row['SpellCheckTweets'].split()
            for x in sentenceSplit:
                embeddingVecForEachWord = embeddings_index.get(x)
                if embeddingVecForEachWord is not None:
                    summedVec += embeddingVecForEachWord
            newVec = [y/len(sentenceSplit) for y in summedVec]
            avgEmbeddedVectors.append(newVec)
        print(avgEmbeddedVectors)
    return avgEmbeddedVectors


readfile()