# -*- coding: utf-8 -*-
"""
Machine Learning project part 1.

(1) Use KNN to impute missing data.
(2) Using KNN, random forests, logistic regressions, and linear SVCs to
    train the data.
(3) Write ROC statistics: Accuracy, precision, specificity, etc. A "positive"
    is taken as the mode of the training outputs ("labels").

Created on Tue Nov 12 13:34:15 2019

@author: Barrett
"""

import numpy as np
import sklearn.metrics as metrics
import Analytics #NEED TO UPLOAD THIS FILE
import ROC_curve #NEED TO UPLOAD THIS FILE
import TimeTracker #NEED TO UPLOAD THIS FILE
from impyute.imputation.cs import fast_knn
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC

KNN_ALG = 0; RAND_FOREST = 1; LOG_REG = 2; LIN_SVC = 3
algorithmsToRun = [0,2] #Set this to the desired algorithms.

MISSING = (1e99, 1e9)
DATA_LISTS = tuple(range(6))
TRUNC_DATA_LISTS = tuple(range(2,6)) #when we want to exclude datasets 1 and 2

analyzeError = True#; analyzeError = False
calcEigs = True; calcEigs = False
loadBigEigs = True; loadBigEigs = False #Set to FALSE on first run (if calcEigs==True)
loadDataFromScratch = True; loadDataFromScratch = False #Set to TRUE on first run
plotROC = True#; plotROC = False
printModeStats = True#; printModeStats = False
saveCleanedData = True; saveCleanedData = False #Set to True for faster future runs

stopwatch = TimeTracker.TimeTracker()
params = ((1,3,5,11,21), #KNN
          (2,5,10,20), #random forest
          tuple(np.power(10, np.arange(-1, 2.5))), #logistic regression
          tuple(np.power(10, np.arange(-1, 2.5)))) #linear SVC
   
print('Loading data')
trainLabel =\
    (np.loadtxt('data/TrainLabel1.txt'), np.loadtxt('data/TrainLabel2.txt'),
     np.loadtxt('data/TrainLabel3.txt'), np.loadtxt('data/TrainLabel4.txt'),
     np.loadtxt('data/TrainLabel5.txt'), np.loadtxt('data/TrainLabel6.txt'))

if loadDataFromScratch:
    trainData =\
        [np.loadtxt('data/TrainData1.txt'), np.loadtxt('data/TrainData2.txt'),
         np.loadtxt('data/TrainData3.txt'), np.loadtxt('data/TrainData4.txt'),
         np.loadtxt('data/TrainData5.txt'), np.loadtxt('data/TrainData6.txt')]

    testData =\
        [np.loadtxt('data/TestData1.txt'), np.loadtxt('data/TestData2.txt'),
         np.loadtxt('data/TestData3.txt', delimiter=','),
         np.loadtxt('data/TestData4.txt'), np.loadtxt('data/TestData5.txt'),
         np.loadtxt('data/TestData6.txt')]
    
    for i in DATA_LISTS:
        for j in MISSING:
            trainData[i][trainData[i] == j] = np.nan
            testData[i][testData[i] == j] = np.nan       
        if np.any(np.isnan(trainData[i])):
            print('Cleaning trainData' + str(i+1))
            trainData[i] = fast_knn(trainData[i], k=5)
        if np.any(np.isnan(testData[i])):
            print('Cleaning testData' + str(i+1))
            testData[i] = fast_knn(testData[i], k=5)
        if saveCleanedData:
            np.save('trainData'+str(i+1), trainData[i])
            np.save('testData'+str(i+1), testData[i])
else:
    trainData = []; testData = []
    for i in DATA_LISTS:
        trainData.append(np.load('trainData' + str(i+1) + '.npy'))
        testData.append(np.load('testData' + str(i+1) + '.npy'))
N_data = len(trainData)

modeID = []
for i in DATA_LISTS:
    if printModeStats:
        print('    Portion of trainLabel' + str(i+1) + ' == mode:',
              round(mode(trainLabel[i])[1][0]/len(trainLabel[i]),3))
    modeID.append(np.where(np.unique(trainLabel[i]) == mode(trainLabel[i])[0][0])[0][0])

'''Set up the analyses.
Dimensions: Algorithm, dataset, algorithm parameters'''
if analyzeError:
    accuracy = len(algorithmsToRun)*[None]
    precision = accuracy.copy()
    sensitivity = accuracy.copy()
    specificity = accuracy.copy()
    F1_score = accuracy.copy()
    FP_array = accuracy.copy()
    TP_array = accuracy.copy()
    for i, run in enumerate(algorithmsToRun):
        accuracy[i] = np.zeros([N_data, len(params[run])])
        precision[i] = accuracy[i].copy()
        sensitivity[i] = accuracy[i].copy()
        specificity[i] = accuracy[i].copy()
        F1_score[i] = accuracy[i].copy()
        FP_array[i] = accuracy[i].copy()
        TP_array[i] = accuracy[i].copy()

'''Finds the eigenvalues of the covariance matrices, which are symmetric, so we
can use the np.linalg.eigvalsh method. Eigenvalue calculations pre-sort the
eigenvalues by magnitude. This part is not necessary for the algorithm, but it
is where I show which eigenvalues to use.'''
if calcEigs:
    cov = []
    print('Calculating covariance matrix eigenvalues')
    for j in DATA_LISTS:
        cov.append(np.cov(trainData[j].T, ddof=0))
    if loadBigEigs:
        eigVals = [np.load('eigVals1.npy'), np.load('eigVals2.npy')]
    else:
        print('EigsVals1'); eigVals = [np.linalg.eigvalsh(cov[0])[::-1]]
        print('EigsVals2'); eigVals.append(np.linalg.eigvalsh(cov[1])[::-1])
        np.save('eigVals1', eigVals[0]); np.save('eigVals2', eigVals[1])
    print('Remaining eigvals') #datasets 3-6 have much smaller eigenspaces
    for i in TRUNC_DATA_LISTS: #eigvalsh pre-sorts the eigenvalues.
        eigVals.append(np.linalg.eigvalsh(cov[i])[::-1])
#trims = np.array([149, 99, 13, 72, 5, 1]) #by visual inspection of the eigenvalues
trims = (53, 74, 9, 9, 11, 1) #all except last given

print('Reducing dimensionality of data')
pca_train = []; pca_test = []
for i in DATA_LISTS:
    pca_train.append(PCA(n_components=trims[i]).fit_transform(trainData[i]))
    pca_test.append(PCA(n_components=[trims[i],None][i<2]).fit_transform(testData[i]))
    #Too few features in the first two testData sets

'''KNN(k=1) perfectly classifies the data in all but the 3rd training
dataset, and it is very close there. This feels like significant overfitting,
so I am excluding k=1 from the final analysis. Also, k=2 was recommended, but
this could create a lot of ambiguous classifications, so I am setting the
second lowest value of k to 3 instead
'''
for i, run in enumerate(algorithmsToRun):
    print('Running', ['KNN', 'random forests', 'logistic regression',
                      'linear SVC'][run])
    useReducedData = run not in (KNN_ALG, RAND_FOREST)
    n = params[run]
    M = len(n)
    data_sets = list(DATA_LISTS)
#    y_test = []
#    for ell, j in enumerate(DATA_LISTS):
#        y_test.append(np.zeros([testData[j].shape[0], M]))
#        ROC_legend.append('Dataset ' + str(i+1))
    for ell, i_param in enumerate(n):    
#        if run == LOG_REG:
        penalty = 'l1'; solver = 'saga' #for logistic regression
#            penalty = ['l2','l1','none'][i//len(c)]
#            solver = ['newton-cg','saga'][penalty == 'l1']
        for j in data_sets:
            machine = [KNN(n_neighbors=i_param),
                RFC(n_estimators=i_param, random_state=0),
                LogReg(penalty=penalty, C=i_param, solver=solver,
                       max_iter=1e6, multi_class='multinomial',
                       random_state=0),
                LinearSVC(C=i_param, dual=True, max_iter=1e5, random_state=0)][run]
            machine.fit([pca_train[j], trainData[j]][useReducedData], trainLabel[j])
#            y_test[j][:,ell] = KNN_array[j].predict(testData[j])            
            if analyzeError:
                tp, fp, fn, tn = Analytics.reduce(metrics.confusion_matrix(trainLabel[j],
                    machine.predict([pca_train[j], trainData[j]][useReducedData])), modeID[j]).flatten()
                tot = tp+fp+fn+tn
                TP_array[i][j,ell] = tp/tot
                FP_array[i][j,ell] = fp/tot
                accuracy[i][j,ell] = Analytics.accuracy(tp, fp, tn, fn)
                sensitivity[i][j,ell] = Analytics.sensitivity(tp, fp, tn, fn)
                specificity[i][j,ell] = Analytics.specificity(tp, fp, tn, fn)
                precision[i][j,ell] = Analytics.precision(tp, fp, tn, fn)
                F1_score[i][j,ell] = Analytics.F1_score(tp, fp, tn, fn)
        print('    ' + str(ell+1) + ' of ' + str(M) + ' complete')
if analyzeError and plotROC:
    for i, run in enumerate(algorithmsToRun): #Waits to show graphs until after analyses
        for j in data_sets:
            sorter = FP_array[i][j].argsort() #x coord on ROC curve
            TP_array[i][j] = TP_array[i][j][sorter]
            FP_array[i][j] = FP_array[i][j][sorter]
            accuracy[i][j] = accuracy[i][j][sorter]
            sensitivity[i][j] = sensitivity[i][j][sorter]
            specificity[i][j] = specificity[i][j][sorter]
            precision[i][j] = precision[i][j][sorter]
            F1_score[i][j] = F1_score[i][j][sorter]
        ROC_curve.plot_ROC(FP_array[i][data_sets], TP_array[i][data_sets],\
           True, True, [' (KNN)', ' (Random forest)', ' (Logistic regression)',\
           ' (Linear SVC)'][run], 1+np.array(data_sets))
#    if saveYs:
#        for ell, curr_dataID in enumerate(DATA_LISTS):
#            np.save('y_test' + str(curr_dataID+1), y_test[ell])
print(stopwatch.getElapsedTime())