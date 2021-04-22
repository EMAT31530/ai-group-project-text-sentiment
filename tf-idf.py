# This short script evaluate how relevant a word is to the tweet in a collection of tweets.
# The script requires the data pre processing script to be run as it reads the cleaned dataset of tweets
# and applies scikit-learn's tf-idf on it
# Term Frequency Inverse Document Frequency computes word frequencies in tweets and assigns relevance scores
# to all words in all tweets by how relevant that particular word is to the tweet it came from.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
data = pd.read_csv('dataset_cleaned_tweets.csv')
dataset = data['SpellCheckTweets']

tfIdfVectorizer=TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(dataset)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print(df.head(25))
