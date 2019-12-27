# -*- coding: utf-8 -*-
"""
Methods for analyzing the receiver operating characteristics of the data.

Created on Tue Nov 19 15:36:44 2019

@author: Barrett
"""

import numpy as np

accuracy = lambda tp, fp, tn, fn: (tp+tn)/(tp+fp+fn+tn)
sensitivity = lambda tp, fp, tn, fn: tp/(tp+fn)
specificity = lambda tp, fp, tn, fn: tn/(tn+fp)
recall = lambda tp, fp, tn, fn: sensitivity(tp, fp, tn, fn) #same as sensitivity!!
F1_score = lambda tp, fp, tn, fn: 2*tp/(2*tp + fp + fn)

def precision (tp, fp, tn, fn):
    if tp+fp == 0:
        return np.nan #avoids dividing by zero, which is common with these data
    return tp/(tp+fp)

'''Reduces an nXn confusion matrix to a 2X2 binary one. critIndex is the row and
column index of the "positive" value.

True positives: Only the entry in question.
False positives: Sum of all other values in that row.
False negatives: Sum of all other values in that column.
True negatives: Sum of every other value in the matrix.'''
def reduce(X, critIndex):
    confusion = np.zeros([2,2], dtype=np.uint32)
    confusion[0,0] = X[critIndex, critIndex] #TPs
    confusion[1,0] = np.sum(X[critIndex,:]) - confusion[0,0] #FPs
    confusion[0,1] = np.sum(X[:,critIndex]) - confusion[0,0] #FNs
    confusion[1,1] = np.sum(X) - np.sum(confusion) #TNs
    return confusion

'''Inspects the training labels.'''
def main():    
    trainLabel1 = np.loadtxt('Dataset/TrainLabel1.txt')
    trainLabel2 = np.loadtxt('Dataset/TrainLabel2.txt')
    trainLabel3 = np.loadtxt('Dataset/TrainLabel3.txt')
    trainLabel4 = np.loadtxt('Dataset/TrainLabel4.txt')
    trainLabel5 = np.loadtxt('Dataset/TrainLabel5.txt')
    trainLabel6 = np.loadtxt('Dataset/TrainLabel6.txt')
    
    print(np.unique(trainLabel1)) #1..5
    print(np.unique(trainLabel2)) #1..11
    print(np.unique(trainLabel3)) #1..9
    print(np.unique(trainLabel4)) #1..9
    print(np.unique(trainLabel5)) #3..8
    print(np.unique(trainLabel6))
    
    #This part is for debugging
    from sklearn.metrics import confusion_matrix
    true = [2,0,2,2,0,1]
    pred = [0,0,2,2,0,2]
    confusion = confusion_matrix(true, pred)
    print('Reduced--TPs, FPs, FNs, TNs:', reduce(confusion, 1).flatten()) #[0,0,1,5]