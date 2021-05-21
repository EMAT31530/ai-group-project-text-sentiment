import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import re
from io import StringIO
import operator

def file_ext_to_list(empty_list, ext):
    for root, dirs_list, files_list in os.walk(dir):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == ext:
                empty_list.append(file_name)

# get log file in list to iterate over later
#/Users/vedangjoshi/PycharmProjects/MDM3_Phase_C/Accelerometer_Data/23Apr2021
# dir is directory where script is run from; also the directory which contains the log files
dir = '/Users/vedangjoshi/PycharmProjects/IntroAI/pkl_lists'
pkl_extension = '.pkl'

pkl_file_name_list = []
file_ext_to_list(pkl_file_name_list, pkl_extension)

classification_list = []
for i in pkl_file_name_list:
    if 'classification' in i:
        classification_list.append(i)


def classification_report_to_dataframe(str_representation_of_report):
    split_string = [x.split(' ') for x in str_representation_of_report.split('\n')]
    column_names = ['']+[x for x in split_string[0] if x!='']
    values = []
    for table_row in split_string[1:-1]:
        table_row = [value for value in table_row if value!='']
        if table_row!=[]:
            values.append(table_row)
    for i in values:
        for j in range(len(i)):
            if i[1] == 'avg':
                i[0:2] = [' '.join(i[0:2])]
            if len(i) == 3:
                i.insert(1,np.nan)
                i.insert(2, np.nan)
            else:
                pass
    report_to_df = pd.DataFrame(data=values, columns=column_names)
    return report_to_df

batch_size_list = []
lr_list = []
dataframe_list = []
for i in classification_list:
    batch_size_plus_lr = i.split("batch_", 1)[1]
    lr_with_extension = batch_size_plus_lr.split("_lr_", 1)[1]
    batch_size = batch_size_plus_lr.split("_lr_", 1)[0]
    lr = lr_with_extension.split(".pkl", 1)[0]
    batch_size_list.append(batch_size)
    lr_list.append(lr)
    with open(dir+'/'+i, 'rb') as f:
        classification_rep = pickle.load(f)
    report_df = classification_report_to_dataframe(classification_rep)
    dataframe_list.append(report_df)


batch_size_list = [int(i) for i in batch_size_list]
lr_list = [float(i) for i in lr_list]
with open('pickledlist.pkl', 'rb') as f:
    pickled_list = pickle.load(f)

nested_list = [list(l) for l in zip(batch_size_list, lr_list)]

weighted_avg_list = []
for i in range(len(dataframe_list)):
    weighted_avg_list.append(dataframe_list[i]['f1-score'][4])

weighted_avg_list = [float(i) for i in weighted_avg_list]
print(weighted_avg_list)
print(batch_size_list)
print(lr_list)

zipped_list = list(zip(weighted_avg_list,batch_size_list,lr_list))
res = sorted(zipped_list, key = operator.itemgetter(0))
print(res)

sns.set()
f, axes = plt.subplots(2, 1, sharey='all')
f.tight_layout()
sns.barplot(x=batch_size_list, y=weighted_avg_list, palette="Blues_d", ax=axes[0])
axes[0].set_title('BERT: Batch size vs Model accuracy')
axes[0].set_xlabel('Batch sizes')
axes[0].set_ylabel('Model Accuracy (%)')
sns.barplot(x=lr_list, y=weighted_avg_list, palette="Blues_d", ax=axes[1])
axes[1].set_title('BERT: Learning Rates vs Model accuracy')
axes[1].set_xlabel('Learning rates')
axes[1].set_ylabel('Model Accuracy (%)')
plt.show()
