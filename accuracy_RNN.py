import pandas as pd

testCSV = pd.read_csv('TestSample.csv')
testPred = pd.read_csv('TestPredictions_WORD2VEC.csv')

concatDf = pd.concat([testCSV, testPred], axis=1)
concatDf = concatDf.replace(-1,0)


def Check(df):
   if df['Sentiment']== df['Prediction']:
      return "True"
   else:
      return "False"

concatDf['result'] = concatDf.apply(Check, axis=1)

print('Accuracy: %f'%(concatDf['result'].value_counts(normalize=True)[0] * 100)+'%')
