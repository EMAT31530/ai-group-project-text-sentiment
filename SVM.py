#Import statements
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import plot_precision_recall_curve
from sklearn.svm import LinearSVC, SVC
from matplotlib import pyplot as plt
import pandas as pd
import random
pd.options.display.max_columns = None

#Open files using with statement
with open('train_neg_full.txt') as f:
    negative = f.readlines()

with open('train_pos_full.txt') as f:
    positive = f.readlines()

#Sampling a random 10000 lines from files so program runs quicker while testing
negative = random.sample(negative, 1000)
positive = random.sample(positive, 1000)


X = negative + positive
y = ([0] * len(negative)) + ([1] * len(positive))

#Splitting all data into 20% test and 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


count_vectorizor = CountVectorizer(max_features=100)
X_train = count_vectorizor.fit_transform(X_train).toarray()
X_test = count_vectorizor.transform(X_test).toarray()

clf = GridSearchCV(SVC(), {
    'C': [0.1, 1, 10],
    #'kernel': ['linear', 'poly', 'rbf']
})
clf.fit(X_train, y_train)

print("Training accuracy", clf.score(X_train, y_train))
print("Test accuracy", clf.score(X_test, y_test))
print(pd.DataFrame(clf.cv_results_))
plot_precision_recall_curve(clf, X_test, y_test)
plt.show()
print()
