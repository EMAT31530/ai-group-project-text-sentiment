import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split

dataset = pd.read_csv('cleaned_tweets.csv', encoding = 'utf-8')
# split into train test sets
train, test = train_test_split(dataset, test_size=0.2)


x_train = train.loc[:,'Tweet']
y_train = train.loc[:,'Sentiment']
x_test = test.loc[:,'Tweet']
y_test = test.loc[:,'Sentiment']


count_vectorizor = CountVectorizer(max_features=100)
x_train = count_vectorizor.fit_transform(x_train).toarray()
x_test = count_vectorizor.transform(x_test).toarray()

clf = RandomForestClassifier(max_leaf_nodes=2)
clf.fit(x_train, y_train)
print("Training accuracy", clf.score(x_train, y_train))
print("Test accuracy", clf.score(x_test, y_test))

