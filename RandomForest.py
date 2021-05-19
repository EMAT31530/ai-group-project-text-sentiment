import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

dataset = pd.read_csv('cleaned_tweets.csv', encoding = 'utf-8')

print(dataset.describe())
dataset_dp = dataset.dropna()


train, test = train_test_split(dataset_dp, test_size=0.2)


x_train = train.loc[:,'SpellCheckTweets']
y_train = train.loc[:,'Sentiment']
x_test = test.loc[:,'SpellCheckTweets']
y_test = test.loc[:,'Sentiment']


count_vectorizor = CountVectorizer(max_features=500)
x_train = count_vectorizor.fit_transform(x_train).toarray()
x_test = count_vectorizor.transform(x_test).toarray()


clf = RandomForestClassifier(n_estimators=50, max_leaf_nodes=50, max_depth=40)
clf.fit(x_train, y_train)
print("Training accuracy", clf.score(x_train, y_train))
print("Test accuracy", clf.score(x_test, y_test))

