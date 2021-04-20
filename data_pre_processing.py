#!/usr/bin/python

'''
Install tweepy:
$ git clone https://github.com/tweepy/tweepy.git
$ cd tweepy
$ pip install
Install pandas:
$ pip install pandas
'''

# Import the libraries

import string

import re
import nltk
import pickle
import contractions
import random
import pandas as pd
from nltk import NaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('treebank')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('twitter_samples')

positive_tweets = twitter_samples.strings('positive_tweets.json')
#print(positive_tweets)
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')

# dictionary_slang.csv must be in the same folder as the .py file
slang_dict_df = pd.read_csv('dictionary_slang.csv', usecols=['Abbrv', 'Full'])
slang_dict = slang_dict_df.set_index('Abbrv')['Full'].to_dict()

def data_clean_function(sentimentTweetList):
    # FullTrain.csv must be in the same folder as the .py file
    tweetsDataframe = pd.DataFrame(sentimentTweetList,columns=['Tweet'])
    tweetsDataframe = tweetsDataframe[:100]
        #pd.read_csv('FullTrain.csv', nrows=5)

    # Uncomment the lines below to get the dataframe showing the tweets obtained from the csv file.
    # print('''
    # Show the tweets obtained from twitter API on a PANDAS dataframe''')
    #print(tweetsDataframe)

    # function to deconstruct contractions ex. 'don't' becomes 'do not' etc.
    def deconstruct_phrase(text):
        deconstructed_word_list = []
        for word in text.split():
            # using contractions.fix to expand the shortened words
            deconstructed_word_list.append(contractions.fix(word))

        expanded_text = ' '.join(deconstructed_word_list)
        return expanded_text

    # Create a function to clean the tweets
    def text_cleaner(text):
        text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
        text = re.sub('<[^>]+>', '', text) # Removing text between <>
        text = re.sub('#', '', text)  # Removing hash tag
        text = re.sub('RT[\s]+', '', text)  # Removing RT
        text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
        text = text.lower()  # make text lowercase
        text = re.sub('\[.*?\]', '', text)  # Remove text in square brackets
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
        text = re.sub('\w*\d\w*', '', text)  # remove words containing numbers
        text = re.sub('[‘’“”…]', '', text)  # remove additional punctuation
        text = re.sub('\n', '', text)  # remove nonsensical text missed earlier
        emoji = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese characters
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"
                           u"\u3030"
                           "]+", re.UNICODE)
        text = emoji.sub(r'', text)
        return text

    def remove_stopwords(x):
        stopList = []
        for word in x.split():
            if word not in (stop):
                stopList.append(word)
        return ' '.join(stopList)

    def spellcheck(df):
        #df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['Stopwords_Removed_Tweets']), axis=1)
        spellList = []
        for i in df['Stopwords_Removed_Tweets']:
            textBlb = TextBlob(i)  # Making our first textblob
            textCorrected = textBlb.correct()
            spellList.append(textCorrected)
        return spellList
    # Function to lemmatise tokenised words. To get a better lemmatisation we first create a parts of speech (PoS) mapping for the
    # tokenised words and the pass along the PoS tags as the second argument to the lemmatiser.

    def lemmatise_tweet(text):
        lemmatizer = WordNetLemmatizer()
        parts_speech_tag = nltk.pos_tag([text])[0][1][0].upper()
        parts_speech_tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        get_pos = parts_speech_tag_dict.get(parts_speech_tag, wordnet.NOUN)
        lemmatise_word_list = []
        for word in text.split():
            lemmatise_word_list.append(lemmatizer.lemmatize(word, get_pos))
        lemmatise_join_txt = ' '.join(lemmatise_word_list)
        return lemmatise_join_txt


    def formalise_slang(text):
        slang_removed_word_list = []
        for word in text.split():
            if word in list(slang_dict.keys()):
                slang_removed_word_list.append(slang_dict[word])
            else:
                slang_removed_word_list.append(word)
        slang_removed_join_txt = ' '.join(slang_removed_word_list)
        return slang_removed_join_txt


    ## Main execution for data pre-processing
    # Create a function to remove the stopwords
    stop = set(stopwords.words('english'))


    # Users may comment the lines below to not print the dataframe after cleaning the tweets obtained from the user.
    # Show the tweets obtained after cleaning the data i.e. after removing stopwords etc)
    #print(tweetsDataframe)


    ###################### Main Execution #########################################################
    # Initialise a number of topics list & a twitter handle list. Users may input
    # custom twitter handles in the file twitter_handles.txt which will be read into the program.

    tweetsDataframe['Slang_Removed_Tweets'] = tweetsDataframe['Tweet'].apply(formalise_slang)
    # Deconstruct and lemmatise phrases:
    tweetsDataframe['Deconstruct_Tweets'] = tweetsDataframe['Slang_Removed_Tweets'].apply(deconstruct_phrase)

    tweetsDataframe['Lemmatise_Tweets'] = tweetsDataframe['Deconstruct_Tweets'].apply(lemmatise_tweet)

    # Clean the tweets and remove stopwords
    tweetsDataframe['First_Clean_Tweets'] = tweetsDataframe['Lemmatise_Tweets'].apply(text_cleaner)
    tweetsDataframe['Stopwords_Removed_Tweets'] = tweetsDataframe['First_Clean_Tweets'].apply(remove_stopwords)
    tweetsDataframe['SpellCheckTweets'] = spellcheck(tweetsDataframe)

    newlist = []

    for i in tweetsDataframe['SpellCheckTweets'].tolist():
        i = str(i)
        newlist.append(i)
    tokenlist = []
    tokenized_sents = [word_tokenize(i) for i in newlist]
    for i in tokenized_sents:
        tokenlist.append(i)

    resultantList = sum(tokenlist, [])
    return resultantList

def get_tweets_for_model(cleaned_tokens_list):
    yield dict([token, True] for token in cleaned_tokens_list)


resultantPositiveList = data_clean_function(positive_tweets)
resultantNegativeList = data_clean_function(negative_tweets)


positiveDict = get_tweets_for_model(resultantPositiveList)
negativeDict = get_tweets_for_model(resultantNegativeList)


print(positiveDict)
print(negativeDict)

positive_dataset = [(tweet_dict, 'Positive') for tweet_dict in positiveDict]
negative_dataset = [(tweet_dict, 'Negative') for tweet_dict in negativeDict]

print(positive_dataset)
print(negative_dataset)

dataset = positive_dataset + negative_dataset
random.shuffle(dataset)
train, test = train_test_split(dataset, test_size=0.3)

#print(trainData[0][0])
## Training a Naive Bayes Classifier after splitting the dataset 70-30 for train and test. Future work will add SVMs and may look at deep 
## learning methods like CNN for sentiment analysis

classifier = NaiveBayesClassifier.train(train)
filename = 'firstmodel.pkl'
pickle.dump(classifier, open(filename, 'wb'))
infile = open('firstmodel.pkl','rb')
model = pickle.load(infile)
