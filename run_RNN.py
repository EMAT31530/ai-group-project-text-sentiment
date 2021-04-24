import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from TextToTensor import TextToTensor
from RNN_model import * 
from embeddings_matrix import *


embed_path = 'Embeddings/practise_embedding.txt'
embed_dim = 100 # must be same size as the embedding dimension 

data = pd.DataFrame()
data = pd.read_csv('Datasets/Smaller Datasets/Train.csv', encoding = 'utf-8')
data.head()

X_train, X_test, y_train, y_test = train_test_split(data.Tweet, data.Sentiment, test_size=0.35, random_state=4)

# Tokenizing the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
num_unique_words = len(tokenizer.word_index)

# Creating the embedding matrix
embedding = Embeddings(embed_path, embed_dim)
embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

# Getting the biggest sentence
max_len = np.max([len(text.split()) for text in X_train])

# Converting to tensor
TextToTensor_instance = TextToTensor(tokenizer=tokenizer,max_len=max_len)
X_train_NN = TextToTensor_instance.string_to_tensor(X_train)


model = Sequential()
model.add(Embedding(input_dim=num_unique_words+1, output_dim=embed_dim, input_length=max_len,weights=[embedding_matrix]))

model.compile('rmsprop', 'mse')
output_array = model.predict(X_train_NN)[0]



results = Pipeline( X_train=X_train, Y_train=y_train, embed_path=embed_path, embed_dim=embed_dim,
X_test=X_test, Y_test= y_test, epochs=1, batch_size=256)


'''good = ['Fire in Vilnius! Where is the fire brigade??? #emergency']

TextToTensor_instance = TextToTensor(
tokenizer=results.tokenizer,
max_len=20
)
# Converting to tensors
good_nn = TextToTensor_instance.string_to_tensor(good)

# Forecasting
p_good = results.model.predict(good_nn)[0][0]'''








