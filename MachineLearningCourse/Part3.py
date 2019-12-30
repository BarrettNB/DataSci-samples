# -*- coding: utf-8 -*-
"""
Machine Learning project part 3 on multi-label data.

(1) Use KNN to impute missing data.
(2) Using KNN, random forests, logistic regressions, and linear SVCs to
    train the data.
(3) Write ROC statistics: Accuracy, precision, specificity, etc. For regressions
    and SVCs, a "positive" is taken as the mode of the training outputs ("labels").

Created on Sun Dec  8 20:38:12 2019

@author: Barrett
"""

import numpy as np
import sklearn.metrics as metrics
import Analytics #NEED TO UPLOAD THIS FILE
import ROC_curve #NEED TO UPLOAD THIS FILE
import TimeTracker #NEED TO UPLOAD THIS FILE
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC

KNN_ALG = 0; RAND_FOREST = 1; LOG_REG = 2; LIN_SVC = 3
algorithmsToRun = [0,1,2,3] #Set this to the desired algorithms.

analyzeError = True#; analyzeError = False
plotROC = True#; plotROC = False
printModeStats = True#; printModeStats = False

stopwatch = TimeTracker.TimeTracker()
params = ((1,3,5,11,21), #KNN
          (2,5,10,20), #random forest
          tuple(np.power(10, np.arange(-1, 2.5))), #logistic regression
          tuple(np.power(10, np.arange(-1, 2.5)))) #linear SVC

print('Loading data')
trainLabel = np.loadtxt('data/MultLabelTrainLabel.txt')
trainData = np.loadtxt('data/MultLabelTrainData.txt')
testData = np.loadtxt('data/MultLabelTestData.txt')
trainSum = trainLabel.sum(axis=1) #needed for logistic regression

if printModeStats:
    print('    Portion of trainLabel == mode:',
          np.round(mode(trainSum)[1][0]/len(trainSum),3))
modeID = np.where(np.unique(trainSum) == mode(trainSum, axis=None)[0][0])[0][0]

'''Set up the analyses.
Dimensions: Algorithm, algorithm parameters'''
if analyzeError:
    accuracy = len(algorithmsToRun)*[None]
    precision = accuracy.copy()
    sensitivity = accuracy.copy()
    specificity = accuracy.copy()
    F1_score = accuracy.copy()
    FP_array = accuracy.copy()
    TP_array = accuracy.copy()
    for i, run in enumerate(algorithmsToRun):
        accuracy[i] = np.zeros(len(params[run]))
        precision[i] = accuracy[i].copy()
        sensitivity[i] = accuracy[i].copy()
        specificity[i] = accuracy[i].copy()
        F1_score[i] = accuracy[i].copy()
        FP_array[i] = accuracy[i].copy()
        TP_array[i] = accuracy[i].copy()

for i, run in enumerate(algorithmsToRun):
    print('Running', ['KNN', 'random forests', 'logistic regression',
                      'linear SVC'][run])
    n = params[run]
    M = len(n)
    sumTheLabels = run in [LOG_REG, LIN_SVC]
    for ell, i_param in enumerate(n):    
        penalty = 'l1'; solver = 'saga' #for logistic regression
#            penalty = ['l2','l1','none'][i//len(c)]
#            solver = ['newton-cg','saga'][penalty == 'l1']       
        machine = [KNN(n_neighbors=i_param),
            RFC(n_estimators=i_param, random_state=0),
            LogReg(penalty=penalty, C=i_param, solver=solver,
                   max_iter=1e6, multi_class='multinomial',
                   random_state=0),
            LinearSVC(C=i_param, dual=True, max_iter=1e5, random_state=0)][run]
        machine.fit(trainData, [trainLabel, trainSum][sumTheLabels])
        if analyzeError:
            confusion = metrics.confusion_matrix([trainLabel,trainSum]\
                [sumTheLabels].flatten(), machine.predict(trainData).flatten())
            if sumTheLabels:
                confusion = Analytics.reduce(confusion, modeID)
            tp, fp, fn, tn = confusion.flatten()
            tot = tp+fp+fn+tn
            TP_array[i][ell] = tp/tot
            FP_array[i][ell] = fp/tot
            accuracy[i][ell] = Analytics.accuracy(tp, fp, tn, fn)
            sensitivity[i][ell] = Analytics.sensitivity(tp, fp, tn, fn)
            specificity[i][ell] = Analytics.specificity(tp, fp, tn, fn)
            precision[i][ell] = Analytics.precision(tp, fp, tn, fn)
            F1_score[i][ell] = Analytics.F1_score(tp, fp, tn, fn)
        print('    ' + str(ell+1) + ' of ' + str(M) + ' complete')
if analyzeError and plotROC:
    for i, run in enumerate(algorithmsToRun): #Waits to show graphs until after analyses 
        sorter = FP_array[i].argsort() #x coord on ROC curve
        TP_array[i] = TP_array[i][sorter]
        FP_array[i] = FP_array[i][sorter]
        accuracy[i] = accuracy[i][sorter]
        sensitivity[i] = sensitivity[i][sorter]
        specificity[i] = specificity[i][sorter]
        precision[i] = precision[i][sorter]
        F1_score[i] = F1_score[i][sorter]
        ROC_curve.plot_ROC(FP_array[i], TP_array[i], showTitle=True, showLegend=False,\
           titleAddition = [' (KNN)', ' (Random forest)', ' (Logistic regression)',\
           ' (Linear SVC)'][run])

print(stopwatch.getElapsedTime())