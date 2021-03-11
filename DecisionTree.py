#Import statements
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import random

#Open files using with statement
with open('train_neg_full.txt') as f:
    negative = f.readlines()

with open('train_pos_full.txt') as f:
    positive = f.readlines()

#Sampling a random 10000 lines from files so program runs quicker while testing
negative = random.sample(negative, 10000)
positive = random.sample(positive, 10000)


X = negative + positive
y = ([0] * len(negative)) + ([1] * len(positive))

#Splitting all data into 20% test and 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


count_vectorizor = CountVectorizer(max_features=100)
X_train = count_vectorizor.fit_transform(X_train).toarray()
X_test = count_vectorizor.transform(X_test).toarray()

clf = DecisionTreeClassifier(max_leaf_nodes=2)
clf.fit(X_train, y_train)
print("Training accuracy", clf.score(X_train, y_train))
print("Test accuracy", clf.score(X_test, y_test))

print()
