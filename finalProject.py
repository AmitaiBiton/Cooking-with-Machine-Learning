import random
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix
from scipy.stats import multivariate_normal
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pdb
import re
import base64
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
from sklearn.metrics import classification_report


def FrequencyCalculationPerCountryPerItem(train):
    """
       First:
       For each country or cuisine for each spice or cooking item we will
        count how many times it appears in all the dishes so that finally we
        will have a data structure that will hold for any country all the
        commonations for a spice or cooking item
       """
    counters = {}
    # for each country :
    for cuisine in train['cuisine'].unique():

        # counter the cooking item
        counters[cuisine] = Counter()
        # true if cuisine == this country
        indices = (train['cuisine'] == cuisine)
        # for each cooking item in this country
        for ingredients in train[indices]['ingredients']:
            counters[cuisine].update(ingredients)
    return counters

def Top10Features(counters):
    top10 = pd.DataFrame([[items[0] for items in counters[cuisine].most_common(10)] for cuisine in counters],
                        index=[cuisine for cuisine in counters],
                        columns=['top{}'.format(i) for i in range(1, 11)])
    return  top10

def Logistic_Regression(X_train, X_test, y_train, y_test , train):
    logistic = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)
    plt.figure(figsize=(10, 10))
    cm = metrics.confusion_matrix(y_test, y_pred)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest')
    plt.colorbar(shrink=0.3)
    cuisines = train['cuisine'].value_counts().index
    tick_marks = np.arange(len(cuisines))
    plt.xticks(tick_marks, cuisines, rotation=90)
    plt.yticks(tick_marks, cuisines)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print(classification_report(y_test, y_pred, target_names=cuisines))


def MaxOnTheta(X_train , y_train , num):

    l = 'l2'
    # find the Two feature that give you the max value theta ,
    # after that you find the first max search for the second
    regression = LogisticRegression(C=0.10, penalty='l2', solver='lbfgs', max_iter=10000)
    regression.fit(X_train, y_train)
    # get the theta
    theta = regression.coef_[0]
    theta =abs(theta)
    features = []
   # find first max
    for i in range(num):
        feature = np.where(theta==np.amax(theta))[0][0]
        theta[feature]=0
        features.append(feature)
        # find the second one
    return features


def get3Classes(df_train):

    class1 = df_train['cuisine'].value_counts().keys()[0]
    class2 = df_train['cuisine'].value_counts().keys()[1]
    class3 = df_train['cuisine'].value_counts().keys()[2]
    class4 = df_train['cuisine'].value_counts().keys()[3]
    class5 = df_train['cuisine'].value_counts().keys()[4]
    class6 = df_train['cuisine'].value_counts().keys()[5]
    train1  = df_train [ df_train.cuisine ==class1]
    train2  = df_train [ df_train.cuisine ==class2]
    train3  = df_train [ df_train.cuisine == class3]
    train4 = df_train[df_train.cuisine == class4]
    train5 = df_train[df_train.cuisine == class5]
    train6 = df_train[df_train.cuisine == class6]
    dfs = [train1 , train2 ,train3 ,train4 , train5 ,train6]
    train = pd.concat(dfs)
    train =shuffle(train)

    return train

def AdaBoost(X_train , X_test , y_train , y_test , train):

    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest')
    plt.colorbar(shrink=0.3)
    cuisines = train['cuisine'].value_counts().index
    tick_marks = np.arange(len(cuisines))
    plt.xticks(tick_marks, cuisines, rotation=90)
    plt.yticks(tick_marks, cuisines)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print(classification_report(y_test, y_pred, target_names=cuisines))

