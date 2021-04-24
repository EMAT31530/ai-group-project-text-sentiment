import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pd.DataFrame()
data = pd.read_csv('Datasets/TweetsTrain.csv', encoding = 'utf-8')
data.head()

X_train, X_test, y_train, y_test = train_test_split(data.Tweet, data.Sentiment, test_size=0.35, random_state=4)

lr = LogisticRegression(penalty='l2', C=.8, random_state=21)
text_classifier = Pipeline([('vectorizer', CountVectorizer(binary=False, stop_words='english', ngram_range=(1,2))), ('tfidf', TfidfTransformer(use_idf=True)),('clf', lr),])
text_classifier.fit(X_train, y_train)


predicted_test = text_classifier.predict(X_test)
predicted_proba_test = text_classifier.predict_proba(X_test)



predicted_train = text_classifier.predict(X_train)

y_train = y_train.astype('category')
print(metrics.classification_report(y_train, predicted_train,labels=y_train.cat.categories.tolist()))

ConMatTrain=metrics.confusion_matrix(y_train, predicted_train)
TrainAccuracy = (ConMatTrain[0,0] + ConMatTrain[1,1]) / (ConMatTrain[0,0] + ConMatTrain[1,1] + ConMatTrain[0,1] + ConMatTrain[1,0])

print('TRAINING CONFUSION MATRIX:')
print(ConMatTrain)
print(f'Training Accuracy = {TrainAccuracy:.2f}')

predicted_test = text_classifier.predict(X_test)

y_test = y_test.astype('category')
print(metrics.classification_report(y_test, predicted_test,labels=y_test.cat.categories.tolist()))

ConMatTest = metrics.confusion_matrix(y_test, predicted_test)
TestAccuracy = (ConMatTest[0,0] + ConMatTest[1,1]) / (ConMatTest[0,0] + ConMatTest[1,1] + ConMatTest[0,1] + ConMatTest[1,0])

print('TESTING CONFUSION MATRIX:')
print(ConMatTest)
print(f'Test Accuracy = {TestAccuracy:.2f}')