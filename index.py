# import kaggle
# from kaggle.api.kaggle_api_extended import KaggleApi

# Load libraries
# from pandas.plotting import scatter_matrix
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.preprocessing import OneHotEncoder

# api = KaggleApi()
# api.authenticate()

# dataset = read_csv(url, names=names, usecols=[0, 7])
# X = array[:,1]
# y = array[:,4]

# print(dataset.groupby('Subreddit').size())

# dataset = pd.read_csv(url, names=names, encoding="utf-8")

# array = dataset.groupby('Subreddit').size()

# array = OneHotEncoder().fit_transform(dataset.values)
# df = array.view(str)
# dd = np.array(arr)
# df = pd.DataFrame(array).drop([1] + [2], axis=1)

# print(df.head())
# np.char.decode(df)


import nltk
import numpy as np
import pandas as pd
from pandas import read_csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


url = "./rdata.csv"
names = [
    'Title',
    'Political_Lean',
    'Score',
    'ID',
    'Subreddit',
    'URL',
    'Num_Of_Comments',
    'Text',
    'Data_Created',
]

dataset = read_csv(url, names=names).values
array = pd.DataFrame(dataset).iloc[:, (0)]

# print(df)

analyzer = SIA()
results = []

for line in array:
    pol_score = analyzer.polarity_scores(line)
    pol_score['Text'] = line
    results.append(pol_score)

    print("{:-<65} {} \n".format(line, str(pol_score)))
# print("{:-<65} {}".format(line, str(pol_score)))
# print(results[:10])
