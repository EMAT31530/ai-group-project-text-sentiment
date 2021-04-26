import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

data1 = pd.DataFrame()
data1 = pd.read_csv('Datasets/Train/Train.csv', encoding = 'utf-8')

data2 = pd.DataFrame()
data2 = pd.read_csv('Datasets/Test/Test.csv', encoding = 'utf-8')

x_train = data1.loc[:,'Tweet']
y_train = data1.loc[:,'Sentiment']
x_test = data2.loc[:,'Tweet']
y_test = data2.loc[:,'Sentiment']


count_vectorizor = CountVectorizer(max_features=100)
x_train = count_vectorizor.fit_transform(x_train).toarray()
x_test = count_vectorizor.transform(x_test).toarray()

clf = DecisionTreeClassifier(max_leaf_nodes=2)
clf.fit(x_train, y_train)
print("Training accuracy", clf.score(x_train, y_train))
print("Test accuracy", clf.score(x_test, y_test))