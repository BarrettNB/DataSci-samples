# -*- coding: utf-8 -*-
"""
Machine Learning project part 2.

Created on Sat Nov 16 11:19:08 2019

@author: Barrett
"""

import numpy as np
import TimeTracker #NEED TO UPLOAD THIS FILE
from impyute.imputation.cs import fast_knn

k_array = np.arange(1,5.5, dtype=np.int)
MISSING = 1e99

stopwatch = TimeTracker.TimeTracker()
print('Loading data')
allImputed = -np.ones([len(k_array), 3])
data = [np.loadtxt('data/MissingData1.txt'),
        np.loadtxt('data/MissingData2.txt'),
        np.loadtxt('data/MissingData3.txt')]

for k_iter, k_value in enumerate(k_array):
    for i in range(3):
        data[i][data[i] == MISSING] = np.nan
        if np.any(np.isnan(data[i])):
            print('Cleaning dataset' + str(i+1))
            data[i] = fast_knn(data[i], k=k_value)
        allImputed[k_iter, i] = not(np.any(np.isnan(data[i])))
    print(k_iter+1, 'of', len(k_array), 'k values complete')

print('\nAll data imputed?', np.all(allImputed).astype(np.bool))
print(stopwatch.getElapsedTime())