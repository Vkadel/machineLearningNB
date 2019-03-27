#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys

from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

#imports
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print(features_test[0])


#########################################################
### your code goes here ###
loop=[0,1,2,3,4,5,6,7,8,9]
for i in loop:
    gnb = GaussianNB(var_smoothing=1e-7)
    pred = gnb.fit(features_train, labels_train).predict(features_test)
    acc=accuracy_score(labels_test,pred)
    print("Accuracy for ",i,": ", acc)
#########################################################


