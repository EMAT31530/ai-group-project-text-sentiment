from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import plot_precision_recall_curve
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import svm


dataset = pd.read_csv('cleaned_tweets.csv', encoding = 'utf-8')
dataset_dp = dataset.dropna() # Removing rows from the dataset that contain empty values


# Splitting data into 20% test and 80% training
train, test = train_test_split(dataset_dp, test_size=0.2, random_state=42)

x_train = train.loc[:,'SpellCheckTweets']
y_train = train.loc[:,'Sentiment']
x_test = test.loc[:,'SpellCheckTweets']
y_test = test.loc[:,'Sentiment']


count_vectorizor = CountVectorizer(max_features=500)
x_train = count_vectorizor.fit_transform(x_train).toarray()
x_test = count_vectorizor.transform(x_test).toarray()


clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train)

print("Training accuracy", clf.score(x_train, y_train))
print("Test accuracy", clf.score(x_test, y_test))
plot_precision_recall_curve(clf, x_test, y_test)
plt.show()

