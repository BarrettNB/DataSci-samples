# -*- coding: utf-8 -*-
"""
Reads DNA/RNA data from a GenBank file and converts it into a numpy array.

Created on Wed Aug 14 09:18:54 2019

@author: Barrett
"""

import numpy as np

IUPAC = ('A','C','G','U','T','R','Y','S','W','K','M','B','D','H','V','N','-')
GATTACA = ('G','A','T/U','C')
eps = 1e-6
'''Probability matrix for nucleotides based on IUPAC nucleotide codes.
Columns are the IUPAC codes, and rows are G-A-T/U-C.
https://www.bioinformatics.org/sms/iupac.html'''
PROBABILITIES = np.array([[0,0,1,0,0, 1/2,0,1/2,0,1/2,0, 1/3,1/3,0,1/3, 1/4, eps],
                          [1,0,0,0,0, 1/2,0,0,1/2,0,1/2, 0,1/3,1/3,1/3, 1/4, eps],
                          [0,0,0,1,1, 0,1/2,0,1/2,1/2,0, 1/3,1/3,1/3,0, 1/4, eps],
                          [0,1,0,0,0, 0,1/2,1/2,0,0,1/2, 1/3,0,1/3,1/3, 1/4, eps]])

'''Get the nucleotide data from a GenBank file'''
def readData(directory, fname):
    with open(directory + fname, encoding="utf8") as f:
        read_data = f.read()
    read_data = read_data.split('>')[1:]
    for i in range(len(read_data)):
        read_data[i] = read_data[i].split('\n', 1)[1].replace('\n','')
    return read_data

'''Converts the nucleotide code sequence to a probability array. Each nucleotide
gets 4 entries. For example, R means A or G, so its GATC probabilities will be
0.5-0.5-0-0.'''
def letters_to_numbers(string):
    samples = len(string)
    nucleotidesPerSample = []
    for i in range(samples):
        if len(string[i]) not in nucleotidesPerSample:
            nucleotidesPerSample.append(len(string[i]))
    sequences = []
    for i in range(len(nucleotidesPerSample)):
        #The -1 multiplier flags any genes that got missed.
        curr_seq = -1*np.ones([samples, 4*nucleotidesPerSample[i]])
        rows_to_cut = []
        for j in range(samples):
            if len(string[j]) == nucleotidesPerSample[i]:
                for k in range(0, 4*nucleotidesPerSample[i], 4):
                    curr_seq[j,k:k+4] = PROBABILITIES[:, IUPAC.index(string[j][int(k/4)])]
            else: #wrong nucleotide length
                rows_to_cut.append(j)
        curr_seq = np.delete(curr_seq, rows_to_cut, axis=0)
        sequences.append(curr_seq)
    return sequences

def numbers_to_CSV(cMarkerIDs, iMarkerIDs):
    import pandas as pd 
    pd.DataFrame(data = np.c_[cMarkerIDs//4, np.array(GATTACA)\
        [np.mod(cMarkerIDs, 4)]], columns=['Position','Base']).to_csv('locations_of_cVDPV_.csv')
    pd.DataFrame(data = np.c_[iMarkerIDs//4, np.array(GATTACA)\
        [np.mod(iMarkerIDs, 4)]], columns=['Position','Base']).to_csv('locations_of_iVDPV.csv')
    print()

def readData_as_numbers(directory, fname):
    return letters_to_numbers(readData(directory, fname))

'''Returns the G/A/T/C count of a sample.'''
def split_by_nucleotide(X, proportions=True):
    p_nuclueotides = X.shape[1]//4
    if p_nuclueotides != X.shape[1]/4:
        raise IndexError('width of data to analyze must be divisible by 4')
    GATC_counts = np.zeros([p_nuclueotides, 4])
    indeces = np.arange(0, 4*(p_nuclueotides-1) + 0.5, 4, dtype=np.int)
    for i in range(4):
        GATC_counts[:,i] = np.sum(X[:,indeces + i], axis=0)
    if proportions:
        GATC_counts = np.divide(GATC_counts.T, np.sum(GATC_counts, axis=1)).T
    return GATC_counts