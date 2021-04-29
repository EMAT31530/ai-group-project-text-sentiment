import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC

data1 = pd.DataFrame()
data1 = pd.read_csv('Datasets/Train/Train.csv', encoding = 'utf-8')

data2 = pd.DataFrame()
data2 = pd.read_csv('Datasets/Test/Test.csv', encoding = 'utf-8')


#X_train, X_test, y_train, y_test = train_test_split(data.Tweet, data.Sentiment, test_size=0.2, random_state=4)
X_train = data1.loc[:,'Tweet']
y_train = data1.loc[:,'Sentiment']
X_test = data2.loc[:,'Tweet']
y_test = data2.loc[:,'Sentiment']

tfidf = TfidfVectorizer(use_idf=True)
classifier = LinearSVC()

clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
clf.fit(X_train, y_train)

# Obtaining the accuracy and confusion matrix when passing train set
predicted_train = clf.predict(X_train)
predicted_proba_train = clf.predict_proba(X_train)
y_train = y_train.astype('category')
print(metrics.classification_report(y_train, predicted_train,labels=y_train.cat.categories.tolist()))
ConMatTrain=metrics.confusion_matrix(y_train, predicted_train)
TrainAccuracy = (ConMatTrain[0,0] + ConMatTrain[1,1]) / (ConMatTrain[0,0] + ConMatTrain[1,1] + ConMatTrain[0,1] + ConMatTrain[1,0])
print('TRAINING CONFUSION MATRIX:')
print(ConMatTrain)
print(f'Training Accuracy = {TrainAccuracy:.2f}')


# Obtaining the accuracy and confusion matrix when passing test set
predicted_test = clf.predict(X_test)
predicted_proba_test = clf.predict_proba(X_test)
y_test = y_test.astype('category')
print(metrics.classification_report(y_test, predicted_test,labels=y_test.cat.categories.tolist()))
ConMatTest = metrics.confusion_matrix(y_test, predicted_test)
TestAccuracy = (ConMatTest[0,0] + ConMatTest[1,1]) / (ConMatTest[0,0] + ConMatTest[1,1] + ConMatTest[0,1] + ConMatTest[1,0])
print('TESTING CONFUSION MATRIX:')
print(ConMatTest)
print(f'Test Accuracy = {TestAccuracy:.2f}')


# Saving the predictions to a csv file
#data2['Predictions'] = predicted_test
#data2.to_csv('log_reg_preds.csv')