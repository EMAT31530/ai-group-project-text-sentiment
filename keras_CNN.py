import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import dill as pickle
import pipreqs

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Conv1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from keras.preprocessing.text import Tokenizer

import sklearn
from numpy import array
from numpy import asarray
from numpy import zeros

tweets = pd.read_csv("TrainData.csv")

tweets.isnull().values.any()


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


X = []
sentences = list(tweets['tweet'])
for sen in sentences:
    X.append(preprocess_text(sen))

y = tweets['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_val = np.asarray(X_val)
y_test = np.asarray(y_test)
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)



tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
glove_file = open('Embeddings/glove.twitter.27B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


def build_keras_base(hidden_layers = [64, 64, 64], dropout_rate = 0,
                     l2_penalty = 0.1, optimizer = 'adam',
                     n_input = 100, n_class = 2):
    model = Sequential()

    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)
    model.add(Activation('softmax'))
    model.add(Conv1D(128, 5, activation='relu', data_format='channels_first'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model


model_keras = KerasClassifier(
    build_fn=build_keras_base
)

early_stop = EarlyStopping(
    monitor='val_loss', min_delta=0.1, patience=5, verbose=0)

callbacks = [early_stop]
keras_fit_params = {
    'callbacks': callbacks,
    'epochs': 200,
    'batch_size': 2048,
    'validation_data': (X_val, y_val),
    'verbose': 0
}

dropout_rate_opts = [0, 0.2, 0.5]
hidden_layers_opts = [[64, 64, 64, 64], [32, 32, 32, 32, 32], [100, 100, 100]]
l2_penalty_opts = [0.01, 0.1, 0.5]
keras_param_options = {
    'hidden_layers': [[64, 64, 64, 64], [32, 32, 32, 32, 32], [100, 100, 100]],
    'dropout_rate': [0, 0.2, 0.5],
    'l2_penalty': [0.01, 0.1, 0.5]
}

rs_keras = RandomizedSearchCV(
    model_keras,
    param_distributions=keras_param_options,
    scoring='neg_log_loss',
    n_iter=3,
    cv=3,
    n_jobs=1,
    verbose=1
)

rs_keras.fit(X_train, y_train, **keras_fit_params)

# filename = 'cnn_pickle.sav'
# pickle.dump(model, open(filename, 'wb'))


print(rs_keras.summary())

print('Best score obtained: {0}'.format(rs_keras.best_score_))
print('Parameters:')
for param, value in rs_keras.best_params_.items():
    print('\t{}: {}'.format(param, value))

# score = model.evaluate(X_test, y_test, verbose=1)

# print("Test Score:", score[0])
# print("Test Accuracy:", score[1])

best_model = rs_keras.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(X_train, y_train)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
