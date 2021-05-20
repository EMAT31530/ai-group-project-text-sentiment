from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

dataset = pd.read_csv('cleaned_tweets.csv', encoding = 'utf-8')

dataset_dp = dataset.dropna()


train, test = train_test_split(dataset_dp, test_size=0.2, random_state=42)


x_train = train.loc[:,'SpellCheckTweets']
y_train = train.loc[:,'Sentiment']
x_test = test.loc[:,'SpellCheckTweets']
y_test = test.loc[:,'Sentiment']


count_vectorizor = CountVectorizer(max_features=500)
x_train = count_vectorizor.fit_transform(x_train).toarray()
x_test = count_vectorizor.transform(x_test).toarray()

clf = DecisionTreeClassifier(max_leaf_nodes=500)
clf.fit(x_train, y_train)
print("Training accuracy", clf.score(x_train, y_train))
print("Test accuracy", clf.score(x_test, y_test))
# Result:
# Training accuracy 0.7315084121054083
# Test accuracy 0.7241180452824512


# Other results:
# print("1")
# clf = DecisionTreeClassifier(max_depth=30)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("2")
# clf = DecisionTreeClassifier(max_depth=40)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("3")
# clf = DecisionTreeClassifier(max_depth=50)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# 1
# Training accuracy 0.6511985056289648
# Test accuracy 0.6349071032770854
# 2
# Training accuracy 0.6836183336258557
# Test accuracy 0.6555926083795101
# 3
# Training accuracy 0.7050184289045458
# Test accuracy 0.6659729709399995
# print("4")
# clf = DecisionTreeClassifier(max_leaf_nodes=60)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("5")
# clf = DecisionTreeClassifier(max_leaf_nodes=70)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("6")
# clf = DecisionTreeClassifier(max_leaf_nodes=80)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("7")
# clf = DecisionTreeClassifier(max_leaf_nodes=90)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("8")
# clf = DecisionTreeClassifier(max_leaf_nodes=60, max_depth=60)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("9")
# clf = DecisionTreeClassifier(max_leaf_nodes=60, max_depth=70)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("10")
# clf = DecisionTreeClassifier(max_leaf_nodes=60, max_depth=80)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("11")
# clf = DecisionTreeClassifier(max_leaf_nodes=70, max_depth=60)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("12")
# clf = DecisionTreeClassifier(max_leaf_nodes=70, max_depth=70)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("13")
# clf = DecisionTreeClassifier(max_leaf_nodes=70, max_depth=80)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# 4
# Training accuracy 0.6677092997016273
# Test accuracy 0.6682546448361457
# 5
# Training accuracy 0.6701978286487977
# Test accuracy 0.6707870521274728
# 6
# Training accuracy 0.6786475440663942
# Test accuracy 0.6793872075821779
# 7
# Training accuracy 0.6835807236165785
# Test accuracy 0.6842263621091693
# 8
# Training accuracy 0.6677092997016273
# Test accuracy 0.6682546448361457
# 9
# Training accuracy 0.6677092997016273
# Test accuracy 0.6682546448361457
# 10
# Training accuracy 0.6677092997016273
# Test accuracy 0.6682546448361457
# 11
# Training accuracy 0.6707557104530752
# Test accuracy 0.6710127121831356
# 12
# Training accuracy 0.6701978286487977
# Test accuracy 0.6707870521274728
# 13
# Training accuracy 0.6701978286487977
# Test accuracy 0.6707870521274728
#
# print("1")
# clf = DecisionTreeClassifier(max_leaf_nodes=100)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("2")
# clf = DecisionTreeClassifier(max_leaf_nodes=110)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("3")
# clf = DecisionTreeClassifier(max_leaf_nodes=120)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("4")
# clf = DecisionTreeClassifier(max_leaf_nodes=130)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("5")
# clf = DecisionTreeClassifier(max_leaf_nodes=140)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("6")
# clf = DecisionTreeClassifier(max_leaf_nodes=150)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("6")
# clf = DecisionTreeClassifier(max_leaf_nodes=300)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# 1
# Training accuracy 0.6855991274477847
# Test accuracy 0.6866835493819422
# 2
# Training accuracy 0.6896547401148359
# Test accuracy 0.6906200636862824
# 3
# Training accuracy 0.6922623674247173
# Test accuracy 0.693578717749417
# 4
# Training accuracy 0.6952648998320087
# Test accuracy 0.6966627385101422
# 5
# Training accuracy 0.7007246195120728
# Test accuracy 0.7020033598274954
# 6
# Training accuracy 0.704228618709726
# Test accuracy 0.70548855402051
# 6
# Training accuracy 0.7220808364466064
# Test accuracy 0.7178998570819648


# print("1")
# clf = DecisionTreeClassifier(max_leaf_nodes=100)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("2")
# clf = DecisionTreeClassifier(max_leaf_nodes=200)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("3")
# clf = DecisionTreeClassifier(max_leaf_nodes=300)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("4")
# clf = DecisionTreeClassifier(max_leaf_nodes=400)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("5")
# clf = DecisionTreeClassifier(max_leaf_nodes=500)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("6")
# clf = DecisionTreeClassifier()
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# 1
# Training accuracy 0.6855991274477847
# Test accuracy 0.6866835493819422
# 2
# Training accuracy 0.711988817290575
# Test accuracy 0.7116565955419603
# 3
# Training accuracy 0.7220808364466064
# Test accuracy 0.7178998570819648
# 4
# Training accuracy 0.7268949176340797
# Test accuracy 0.7203319710152195
# 5
# Training accuracy 0.7315084121054083
# Test accuracy 0.7240929719429331
# 6
# Training accuracy 0.9361946192613394
# Test accuracy 0.7020033598274954
#
# print("1")
# clf = DecisionTreeClassifier(max_leaf_nodes=600)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("2")
# clf = DecisionTreeClassifier(max_leaf_nodes=700)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("3")
# clf = DecisionTreeClassifier(max_leaf_nodes=800)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("4")
# clf = DecisionTreeClassifier(max_leaf_nodes=900)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("5")
# clf = DecisionTreeClassifier(max_leaf_nodes=1000)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("6")
# clf = DecisionTreeClassifier(max_leaf_nodes=2000)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# 1
# Training accuracy 0.7344169194895068
# Test accuracy 0.7253967855978738
# 2
# Training accuracy 0.7381904570869794
# Test accuracy 0.7263997191785974
# 3
# Training accuracy 0.7407542060527041
# Test accuracy 0.7267005992528145
# 4
# Training accuracy 0.7429669282651756
# Test accuracy 0.7270766993455858
# 5
# Training accuracy 0.7457751422912018
# Test accuracy 0.7275781661359476
# 6
# Training accuracy 0.7627059148007923
# Test accuracy 0.7274277260988391

# print("1")
# clf = DecisionTreeClassifier(max_leaf_nodes=600, max_depth=50)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("2")
# clf = DecisionTreeClassifier(max_leaf_nodes=700, max_depth=50)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("3")
# clf = DecisionTreeClassifier(max_leaf_nodes=800, max_depth=50)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("4")
# clf = DecisionTreeClassifier(max_leaf_nodes=900, max_depth=50)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("5")
# clf = DecisionTreeClassifier(max_leaf_nodes=1000, max_depth=50)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
# print("6")
# clf = DecisionTreeClassifier(max_leaf_nodes=2000, max_depth=50)
# clf.fit(x_train, y_train)
# print("Training accuracy", clf.score(x_train, y_train))
# print("Test accuracy", clf.score(x_test, y_test))
#
# 1
# Training accuracy 0.6778451972018154
# Test accuracy 0.6688564049845799
# 2
# Training accuracy 0.679161547526515
# Test accuracy 0.6685805982498809
# 3
# Training accuracy 0.6804465561768172
# Test accuracy 0.6681042047990372
# 4
# Training accuracy 0.6816500764736856
# Test accuracy 0.6683549381942181
# 5
# Training accuracy 0.6828097184263973
# Test accuracy 0.6682044981571096
# 6
# Training accuracy 0.6906576736955595
# Test accuracy 0.6667000977860241
