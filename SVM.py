#Import statements
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import plot_precision_recall_curve,classification_report
from sklearn.svm import LinearSVC, SVC
from matplotlib import pyplot as plt
import pandas as pd
import random
import pickle
import numpy as np
import seaborn as sns
pd.options.display.max_columns = None

cleanedDf = pd.read_csv('dataset_cleaned_tweets.csv', usecols=['Sentiment', 'SpellCheckTweets'])
#print(cleanedDf['Sentiment'].value_counts(normalize = True))
cleanedDf = cleanedDf.dropna()
cleanedDf = cleanedDf[:2000]


X = cleanedDf['SpellCheckTweets']
y = cleanedDf['Sentiment']

#Splitting all data into 20% test and 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


count_vectorizor = CountVectorizer(max_features=100)
X_train = count_vectorizor.fit_transform(X_train).toarray()
X_test = count_vectorizor.transform(X_test).toarray()

clf = GridSearchCV(SVC(), {
    'C': [0.1, 1, 10],
    #'kernel': ['linear', 'poly', 'rbf']
})
model = clf.fit(X_train, y_train)

print("Training accuracy", clf.score(X_train, y_train))
print("Test accuracy", clf.score(X_test, y_test))
print(pd.DataFrame(clf.cv_results_))
YPred = model.predict(X_test)
## Dump precision recall curve into a pickle file
pickle.dump(plot_precision_recall_curve(clf, X_test, y_test), open("precision_recall_curve.pkl", "wb"))
##### Dump Classification report into pickle file
pickle.dump(classification_report(y_test,YPred), open("classification_report_svm.pkl", "wb"))
confusion_matrix = pd.crosstab(y_test, YPred)
plt.show()
print()

confusion_matrix = confusion_matrix.to_numpy()

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.


    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    f = plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    f.savefig("SVM_confusion_matrix.pdf", bbox_inches='tight')


labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Negative', 'Positive']
make_confusion_matrix(confusion_matrix,
                      group_names=labels,
                      categories=categories,
                      cmap='Greens', title='Confusion Matrix: SVM model')

