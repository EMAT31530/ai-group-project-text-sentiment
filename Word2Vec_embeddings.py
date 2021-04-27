import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

data = pd.DataFrame()
data = pd.read_csv('Datasets/TweetsSample.csv', encoding = 'utf-8')
data.head()

#create empty list
tweet_data_list = []

indv_lines = data['Tweet'].values.tolist()
for line in indv_lines:
    
    #create word tokens as well as remove punctuation in one go
    rem_tok_punc = RegexpTokenizer(r'\w+')
    
    tokens = rem_tok_punc.tokenize(line)
    
    #convert the words to lower case
    words = [w.lower() for w in tokens]
    
    #invoke all the English stopwords
    stop_word_list = set(stopwords.words('english'))
    
    #remove stop words
    words = [w for w in words if not w in stop_word_list]
    
    #remove <user> and <url>
    useless = ['user', 'url']
    words = [w for w in words if not w in useless]
    
    #append words in the tweet_data_list list
    tweet_data_list.append(words)



EmbeddingDim = 100

#train Word2Vec model
model = Word2Vec(sentences = tweet_data_list, size = EmbeddingDim, workers = 4, min_count = 1)

#Save word embedding model
model_file = 'Embeddings/Practise_embedding.txt'
model.wv.save_word2vec_format(model_file, binary=False)