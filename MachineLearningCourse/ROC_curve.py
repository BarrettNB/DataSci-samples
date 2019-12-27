# -*- coding: utf-8 -*-
"""
Plots the ROC curve from true positive and false positive arrays.

Created on Sun Dec 15 18:34:46 2019

@author: Barrett
"""

import numpy as np
from matplotlib import pyplot as plt

COLORS = ('r','g','b',r'#ff7700','m','c')
FONTSIZE = 14

'''Plots the ROC curve. dim(FP_array) must equal dim(TP_array), which in turn
must equal 1 or 2. 1st dim (row): #datasets, 2nd dim (col): #points/dataset
Returns the AUC (area under curve) per curve.'''
def plot_ROC(FP_array, TP_array, showTitle=False, showLegend=True,
             titleAddition='', Legend=None):
    if FP_array.shape != TP_array.shape:
        raise IndexError('FP_array and TP_array do not have the same shape')
    if not FP_array.ndim in (1,2):
        raise IndexError('Data must have 1 or 2 dimensions, got ' + str(FP_array.ndim))
    if FP_array.ndim == 1:
        FP_array = FP_array.reshape([1, FP_array.size])
        TP_array = TP_array.reshape([1, TP_array.size])
    rows = FP_array.shape[0]
    if rows > len(COLORS):
        raise ValueError('Only enough colors set for ' + str(len(COLORS)) +\
                         ' datasets, need to code more')
    ORIGIN = np.zeros(rows)
    #Put (0,0) and (1,1) on every curve
    FP_array = np.c_[ORIGIN.reshape([rows,1]), FP_array, (ORIGIN+1).reshape([rows,1])]
    TP_array = np.c_[ORIGIN.reshape([rows,1]), TP_array, (ORIGIN+1).reshape([rows,1])]
    fig, ax = plt.subplots(); frame = plt.gca()
    ticks = np.round(np.linspace(0,1,6), 1) #something odd is happening at 0.6
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    frame.set_xticklabels(ticks, fontsize=FONTSIZE)
    frame.set_yticklabels(ticks, fontsize=FONTSIZE)
    ax.set_title('ROC curve' + titleAddition, fontsize=FONTSIZE)
    ax.set_xlabel('False positive rate', fontsize=FONTSIZE)
    ax.set_ylabel('True positive rate', fontsize=FONTSIZE)
    ax.plot([0,1], [0,1], 'k--', zorder=-2)
    Legend = [rows*[''], Legend][showLegend] #avoids an error
    for i in range(rows):
        ax.plot(FP_array[i], TP_array[i], label=Legend[i], c=COLORS[i])
    if showLegend:
        ax.legend(loc='lower right')

def main(): #for debugging
    linear = np.linspace(.2,.8,5)
    fp = (np.c_[linear, linear]).T
    tp = np.sqrt(fp)
    tp[1] = np.power(tp[1], 0.25)
    plot_ROC(fp, tp, COLORS)

#main()