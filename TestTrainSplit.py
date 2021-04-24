import pandas as pd
import csv
from sklearn.model_selection import train_test_split

# Original data set was in the form of two separate .txt files
# First .txt file contained solely 'positive' tweets
# Second .txt file contained solely 'negative' tweets

''' It was deemed desirable to combine both .txt files into one .csv file, 
with an additional column with the labels, indicating that particular tweet's sentiment.
This made it easier to work with, when cleaning the data and when creating word embeddings '''


############# Convert txt file to csv ##############

#Positive tweets
headers = ['Tweet','Sentiment']
df = pd.read_fwf('Datasets/PosTweets.txt', header = None)
df['new_column'] = '1'
df.to_csv('Datasets/PosTweets.csv', header = headers, index = False)

#Negative tweets
df = pd.read_fwf('Datasets/NegTweets.txt', header = None)
df['new_column'] = '-1'
df.to_csv('Datasets/NegTweets.csv', header = None, index = False)


############### Combining Both Files Together ###############

PosReader = csv.reader(open("Datasets/PosTweets.csv"))
NegReader = csv.reader(open("Datasets/NegTweets.csv"))
f = open("Datasets/Tweets.csv", "w")
writer = csv.writer(f)

for row in PosReader:
    writer.writerow(row)
for row in NegReader:
    writer.writerow(row)
f.close()



# Loading the data into a data frame
data = pd.DataFrame()
data = pd.read_csv('Datasets/Tweets.csv', encoding = 'utf-8')
data.head()


# Splitting the data into train and test set
train,test = train_test_split(data, test_size=0.30, random_state=0)

#save the data
train.to_csv('Datasets/Train/Train.csv',index=False)
test.to_csv('Datasets/Test/Test.csv',index=False)


