import numpy as np
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from TextToTensor import TextToTensor
from RNN_model import * 
from embeddings_matrix import *

# Reading the configuration file

with open("conf.yml", 'r') as file:
    conf = yaml.safe_load(file).get('pipeline')

# Getting the embedding
embed_path = 'Embeddings/prac_embedding.txt'
embed_dim = 100 # must be same size as the embedding dimension

# Reading the data
train = pd.read_csv('Datasets/Train/Train.csv', encoding = 'utf-8')[['Tweet', 'Sentiment']]
test = pd.read_csv('Datasets/Test/Test.csv', encoding = 'utf-8')

# Shuffling the data for the k fold analysis
train = train.sample(frac=1)

# Creating the input for the pipeline
X_train = train['Tweet'].tolist()
Y_train = train['Sentiment'].tolist()

X_test = test['Tweet'].tolist()

if conf.get('k_fold'):
    kfold = KFold(n_splits=5)
    acc = []
    f1 = []
    for train_index, test_index in kfold.split(X_train):
        # Fitting the model and forecasting with a subset of data
        k_results = Pipeline(
            X_train=np.array(X_train)[train_index],
            Y_train=np.array(Y_train)[train_index], 
            embed_path=embed_path,
            embed_dim=embed_dim,
            X_test=np.array(X_train)[test_index],
            Y_test=np.array(Y_train)[test_index],
            max_len=conf.get('max_len'),
            epochs=conf.get('epochs'),
            batch_size=conf.get('batch_size')
        )
        # Saving the accuracy
        acc += [k_results.acc]
        f1 += [k_results.f1]
        print(f'The accuracy score is: {acc[-1]}') 
        print(f'The f1 score is: {f1[-1]}') 
    print(f'Total mean accuracy is: {np.mean(acc)}')
    print(f'Total mean f1 score is: {np.mean(f1)}')

# Running the pipeline with all the data
results = Pipeline(X_train=X_train,Y_train=Y_train, embed_path=embed_path,embed_dim=embed_dim,X_test=X_test,max_len=conf.get('max_len'),epochs=conf.get('epochs'),batch_size=conf.get('batch_size'))



# Saving the predictions
test['prob_is_positive'] = results.yhat
test['Prediction'] = [1 if x > 0.5 else -1 for x in results.yhat]
 
# Saving the predictions to a csv file
if conf.get('save_results'):
    if not os.path.isdir('output'):
        os.mkdir('output')    
    test[['Prediction']].to_csv(f'TestPredictions.csv', index=False)


