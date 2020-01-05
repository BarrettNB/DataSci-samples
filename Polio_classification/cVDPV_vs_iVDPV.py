# -*- coding: utf-8 -*-
"""
Attempts to analyze and predict the differences between cVDPV and iVDPV data.
So far the best results have come from a linear support vector classifier.

Fractions of each base at each position are exported to the CSV files. Compare
results to those from Zhao, Jorba et al. (2017).

Best scores from original, base-4 system: (overall train & test, iVDPV train and test)
    Linear regression: Worthless (too few samples)
    Ridge: Worthless (even with large penalties)
    Lasso: Worthless
    Linear SVC: 0.998, 0.973; 1.000, 0.750 w/ C=2, max_iter=200000, penalty=L1, dual=False
    SVC: 0.993, 0.973; 1.000, 0.750 w/ C=5, gamma=0.01, max_iter=500000
    Neural ntwk: 0.998, 0.966; 0.889, 0.875 w/ tanh, 80x80, alpha = 0 (<= 1e-4)

Best scores from GATC-split system:
    Linear regression: Worthless (too few samples)
    Ridge: Worthless
    Lasso: Worthless
    Logistic regression: 0.993, 0.950; 1.000, 0.875 w/ C=1 (arbitrary), penalty=L1, max_iter=C*1e4
    Linear SVC: 0.998, 0.973; 1.000, 0.875 w/ C=2, max_iter=C*5e6, penalty=L1, dual=False
    SVC: 0.982, 0.953; 0.778, 0.750 w/ C=0.1, gamma=0.01, max_iter=C*1e5
    Neural network: 0.998, 0.973; 0.889, 0.750 w/ tanh, alpha=0.01
    
Reference: Zhao, Jorba et al., Computational and Structural Biotechnology
Journal 15 (2017) 456–462.

Created on Thu Aug 15 15:06:27 2019

@author: Barrett
"""

from sklearn.model_selection import train_test_split
import numpy as np
import Conserved_sequence_charts as CSC
import GenBank_nucleotide_reader as GNR
import Timer
import matplotlib.pyplot as plt
from matplotlib import style
style.use('classic')
GENOME_SIZE = 903

export_to_CSV = True; export_to_CSV = False
showCoeffs = True#; showCoeffs = False
retrain = True#; retrain = False #has to be True on new runs
#showPositionLetters = True#; showPositionLetters = False
showConservedSeqs = True#; showConservedSeqs = False

BINARY, LIN_REG, RIDGE, LASSO, LOG_REG, LINEAR_SVC, NONLIN_SVC, NEURAL\
    = np.arange(0, 7.5, 1, dtype=np.int)
#model = 3
model = LOG_REG + 1

indexDataType = np.int16
runtime = Timer.Timer()
directory = 'D:/CDC/GenBank_accessions/' #location of accession files
cVDPV_array = np.array([GNR.readData_as_numbers(directory, 'cVDPV2.fasta')][0][0])
iVDPV_seq = GNR.readData_as_numbers(directory, 'iVDPV2.fasta')

iVDPV903_array = np.array([iVDPV_seq[0]][0])
#iVDPV900_array = np.array([iVDPV_seq[1]][0])

data = np.r_[cVDPV_array, iVDPV903_array]
target = np.r_[np.ones(cVDPV_array.shape[0]), np.zeros(iVDPV903_array.shape[0])]
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)
print('Method:', end=' ')

#covarianceEigvals = np.linalg.eigvalsh(np.cov(X_train.T))[::-1]

'''Finds a marker if and only if it is always 0 in the I's and 1 in the C's or
vice versa. This is the harshest criterion possible.'''
if retrain and model == BINARY:
    print('Binary (all-or-nothing check)')
    cMarkerIDs = np.intersect1d(np.nonzero(np.min(cVDPV_array,axis=0) == 1.)[0],\
                        np.nonzero(np.max(iVDPV903_array,axis=0) == 0.)[0])
    iMarkerIDs = np.intersect1d(np.nonzero(np.min(iVDPV903_array,axis=0) == 1.)[0],\
                        np.nonzero(np.max(cVDPV_array,axis=0) == 0.)[0])
    print('cMarkerID count:', np.size(cMarkerIDs))
    print('iMarkerID count:', np.size(iMarkerIDs))
    '''Relaxes the criteria to allow probabilities <1 but always >0.'''
    cSoftMarkerIDs = np.intersect1d(np.nonzero(np.min(cVDPV_array,axis=0) > 0.)[0],\
                        np.nonzero(np.max(iVDPV903_array,axis=0) < 1.)[0])
    iSoftMarkerIDs = np.intersect1d(np.nonzero(np.min(iVDPV903_array,axis=0) > 0.)[0],\
                        np.nonzero(np.max(cVDPV_array,axis=0) < 1.)[0])
    print('cSoftMarkerID count:', np.size(cSoftMarkerIDs))
    print('iSoftMarkerIDs count:', np.size(iSoftMarkerIDs))

if retrain and model in [LIN_REG, RIDGE, LASSO]:
    import sklearn.linear_model as slm
    print(['Linear regression','Ridge regression','Lasso regression'][model-1])
    machine = [slm.LinearRegression(), slm.Ridge(alpha=5e4), slm.Lasso(alpha=1e5)][model-1]
    machine.fit(X_train, y_train)

if retrain and model == LOG_REG:
    from sklearn.linear_model import LogisticRegression
    print('Logistic regression')
    C, iterMultiplier = 1, 10000
    penalty = 'l1'; solver = 'liblinear'
#    penalty = 'l2'; solver = 'newton-cg'
    machine = LogisticRegression(penalty=penalty, solver=solver,\
         class_weight='balanced', max_iter=int(max(1000, C*iterMultiplier)),\
         random_state=0).fit(X_train, y_train)

if retrain and model == LINEAR_SVC: #C, r_s = 2.0, 3; coeffs = 50
    print('Linear SVM')
    saveasCSV = True; saveasCSV = False
    if retrain:
        from sklearn.svm import LinearSVC
        C, iterMultiplier = 2., 5000000
        machine = LinearSVC(C=C, penalty='l1', dual=False, max_iter=int(C*iterMultiplier),\
            class_weight='balanced', random_state=0).fit(X_train, y_train)

if model in [LOG_REG, LINEAR_SVC]:
    coef_raw = machine.coef_[0]
    n_nonzeros = np.count_nonzero(coef_raw)
    print('Nonzero coefficients:', n_nonzeros)
    coef_indices = np.nonzero(coef_raw)[0]
    coef_series = coef_raw[coef_indices]
    coef_series_sorter = np.argsort(np.abs(coef_series))[::-1]
    c = min(25, n_nonzeros) #determined by inspection
    positions = (coef_indices[coef_series_sorter])[:c]
    coef_trimmed = coef_series[coef_series_sorter][:c]
    
    if showCoeffs:
        '''Negative coefficient: Weighted towards i
        Positive coefficient: Weighted towards c'''
        textLabelSize = 30#; latexTickSize = 30
        for i in [1]: #0 for raw array positions, 1 for G/A/T/C+position
            fig, ax = plt.subplots(figsize=(15,6))
            xTicks = np.arange(c-0.5, dtype=indexDataType)
            ax.set_xticks(xTicks)
            xTickLabels = ([positions, 1+positions//4][i]).astype(str)
            if i==1:
                xTickLabels = np.char.add(' ', xTickLabels)
                xTickLabels = np.char.add(np.array(GNR.GATTACA)[np.mod\
                    (positions, 4)], xTickLabels)
            ax.set_xticklabels(xTickLabels, rotation=45)
            ax.axhline(0,0,xTicks[-1], c='k')
            ax.set_xlabel(['Array index', 'Position'][i], fontsize=textLabelSize)
            ax.set_ylabel('Coefficient weight', fontsize=textLabelSize)
            ax.bar(xTicks, coef_trimmed, color=['b','r'][i])
            plt.tight_layout()
        print('Nucleotides plotted:', len(coef_trimmed))

if showConservedSeqs:
    splits_iVDPV = GNR.split_by_nucleotide(X_train[y_train==0])
    splits_cVDPV = GNR.split_by_nucleotide(X_train[y_train==1])    
    CSC.sequence_bar_chart(splits_iVDPV, splits_cVDPV, positions[:c]//4)
    if export_to_CSV:
        import pandas as pd
        pd.DataFrame(splits_iVDPV, columns=GNR.GATTACA).to_csv('splits_iVDPV.csv')
        pd.DataFrame(splits_cVDPV, columns=GNR.GATTACA).to_csv('splits_cVDPV.csv')
        
if retrain and model in [NONLIN_SVC, NEURAL]:
    print(['SVM', 'Neural network'][model-NONLIN_SVC])
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    iterMultiplier = 100000
    layer_sizes = [GENOME_SIZE*4,200,80]
    machine = [SVC(C=2, gamma=0.01, max_iter=int(C*iterMultiplier),\
              class_weight='balanced', probability=True),\
              MLPClassifier(solver='lbfgs', alpha=0.01, activation='tanh', random_state=0,\
              hidden_layer_sizes=layer_sizes[1:])][model-NONLIN_SVC]
    machine.fit(X_train, y_train)
    if retrain and model == NEURAL and showCoeffs:
        vmin = min(np.min(machine.coefs_[0]), np.min(machine.coefs_[1]))
        vmax = max(np.max(machine.coefs_[0]), np.max(machine.coefs_[1]))
        fig, axarr = plt.subplots(1, 2, figsize=(13, 6))
        imgs = []
        for i in range(len(layer_sizes)-1):
            imgs.append(axarr[i].imshow(machine.coefs_[i],  interpolation='none',\
               cmap='viridis', aspect=layer_sizes[i+1]/layer_sizes[i],\
               vmin=vmin, vmax=vmax))
            axarr[i].set_xlabel('Columns in ' + ['1st','2nd'][i] + ' weight matrix')
            axarr[i].set_ylabel(['Input feature', 'Columns in 1st weight matrix'][i])
            axarr[i].set_title(['1st','2nd'][i] + ' matrix')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(imgs[0], cax=cbar_ax)

if model != BINARY:
    print('Overall training set score: {:.3f}'.format(machine.score(X_train, y_train)))
    print('Overall test set score: {:.3f}'.format(machine.score(X_test, y_test)))
    if retrain and model in [RIDGE, LASSO]:
        print('Number of features used:', np.sum(machine.coef_ != 0))
    print('cVDPV training set score: {:.3f}'.format(machine.score(X_train[y_train==1.,:], y_train[y_train==1])))
    print('cVDPV test set score: {:.3f}'.format(machine.score(X_test[y_test==1.,:], y_test[y_test==1])))
    print('iVDPV training set score: {:.3f}'.format(machine.score(X_train[y_train==0.,:], y_train[y_train==0])))
    print('iVDPV test set score: {:.3f}'.format(machine.score(X_test[y_test==0.,:], y_test[y_test==0])))
    print('Predicted iVDPV\'s (should all be 0\'s):', machine.predict(X_test[y_test==0.,:]))
if model in [LOG_REG, NONLIN_SVC, NEURAL]:
    print('Confidence for test iVDPV: ' + str(np.round(np.max\
        (machine.predict_proba(X_test[y_test==0.,:]), axis=1), 4)))
if model == LINEAR_SVC:
    print('Predicted outputs for test iVDPV:\n', np.round(np.dot(\
            X_test[y_test==0.,:], coef_raw), 4))
if model in [LOG_REG, LINEAR_SVC]:
    print('Output for iVDPV[2]:', np.dot(coef_raw, X_test[y_test==0.,:][2]))
    print("Fractional I's used: " + str(len(np.nonzero(np.all([iVDPV903_array\
        [:,coef_indices] > 0., iVDPV903_array[:,coef_indices] < 1.], axis=0))[0])))
    print("Fractional C's used: " + str(len(np.nonzero(np.all([cVDPV_array\
        [:,coef_indices] > 0., cVDPV_array[:,coef_indices] < 1.], axis=0))[0])))

print(runtime.getElapsedTime())