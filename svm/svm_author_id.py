#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.svm import SVC
classifier = SVC(kernel='linear')

#shrink the data set so that it doesn't take so long (cubic)
#print "running on shrunk data set..."
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

# print "\nSVC kernel=linear"
# classifier.fit(features_train, labels_train)
# classifier.predict(features_test)
# print "accuracy:", classifier.score(features_test, labels_test)
#
# print "\nSVC kernel=rbf, C=1.0"
# classifier = SVC(C=1.0, kernel='rbf') #more complex kernel than linear decision boundary kernel
# classifier.fit(features_train, labels_train)
# classifier.predict(features_test)
# print "accuracy:", classifier.score(features_test, labels_test)
#
# print "\nSVC kernel=rbf, C=10"
# classifier = SVC(C=10, kernel='rbf') #more complex kernel than linear decision boundary kernel
# classifier.fit(features_train, labels_train)
# classifier.predict(features_test)
# print "accuracy:", classifier.score(features_test, labels_test)
#
# print "\nSVC kernel=rbf, C=100"
# classifier = SVC(C=100, kernel='rbf') #more complex kernel than linear decision boundary kernel
# classifier.fit(features_train, labels_train)
# classifier.predict(features_test)
# print "accuracy:", classifier.score(features_test, labels_test)
#
# print "\nSVC kernel=rbf, C=1000"
# classifier = SVC(C=1000, kernel='rbf') #more complex kernel than linear decision boundary kernel
# classifier.fit(features_train, labels_train)
# classifier.predict(features_test)
# print "accuracy:", classifier.score(features_test, labels_test)

print "\nSVC kernel=rbf, C=10000" #higher C corresponds to more complex decision boundary
classifier = SVC(C=10000, kernel='rbf') #more complex kernel than linear decision boundary kernel
classifier.fit(features_train, labels_train)
pred = classifier.predict(features_test)
print "accuracy:", classifier.score(features_test, labels_test)
print "email at index 10 is predicted to be label:", pred[10]
print "email at index 26 is predicted to be label:", pred[26]
print "email at index 50 is predicted to be label:", pred[50]

import numpy as np
y = np.array(pred)
#count number of occurences of emails from Sarah (index 0) and Chris (index 1)
num_Sara = (y==0).sum()
num_Chris = (y==1).sum()
print "emails written by Sara:", num_Sara
print "emails written by Chris:", num_Chris

#########################################################