#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys

import time
sys.path.append("../tools/")
from email_preprocess import preprocess

#imports
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels


#########################################################
    ### your code goes here ###
sample_size_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
##Loop to change sample Size
for j in sample_size_list:
    features_train, features_test, labels_train, labels_test = preprocess("../tools/word_data_unix.pkl","../tools/email_authors.pkl",j)
    loop = [0,1,2,3,4,5,6,7,8,9]
    print("Test sample Size:",features_train.size)
    ##Loop to change the var_smoothing
    for i in loop:
        num=1/(10)**i
        gnb = GaussianNB(var_smoothing=num)
        time0=time.time()
        pred = gnb.fit(features_train, labels_train).predict(features_test)
        time1=time.time()
        acc=accuracy_score(labels_test,pred)
        print("Test sample_Size: ",j," Accuracy for ",num,": ", acc,"Ellapsed time: ",time1-time0)
        i=i+1
    j=1+1
#########################################################


