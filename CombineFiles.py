import pandas as pd
import csv

# Original data set was in the form of two separate .txt files
# First .txt file contained solely 'positive' tweets
# Second .txt file contained solely 'negative' tweets

''' It was deemed desirable to combine both .txt files into one .csv file, 
with an additional column with the labels, indicating that particular tweet's sentiment.
This made it easier to work with, when cleaning the data and when creating word embeddings '''


############# Convert txt file to csv ##############

#Positive tweets
headers = ['Tweet','Sentiment']
df = pd.read_fwf('Datasets/train_pos_full.txt')
df['new_column'] = '1'
df.to_csv('Datasets/TrainPosFull.csv', header = headers, index = False)

#Negative tweets
df = pd.read_fwf('Datasets/train_neg_full.txt')
df['new_column'] = '-1'
df.to_csv('Datasets/TrainNegFull.csv', header = None, index = False)


############### Combining Both Files Together ###############

PosReader = csv.reader(open("Datasets/TrainPosFull.csv"))
NegReader = csv.reader(open("Datasets/TrainNegFull.csv"))
f = open("Datasets/TweetsTrain.csv", "w")
writer = csv.writer(f)

for row in PosReader:
    writer.writerow(row)
for row in NegReader:
    writer.writerow(row)
f.close()