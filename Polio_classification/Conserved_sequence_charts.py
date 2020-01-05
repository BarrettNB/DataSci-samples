# -*- coding: utf-8 -*-
"""
Graphs that shows G-A-T/U-C conservation.

Created on Fri Jan  3 13:00:40 2020

@author: Barrett
"""

import matplotlib.pyplot as plt
import numpy as np

GATTACA = ('G', 'A', 'T', 'C')
COLORS = ('#FFB000', 'g', 'r', 'b')

def sequence_bar_chart(X1, X2, positions):
    textLabelSize = 30#; latexTickSize = 30
    fig, ax = plt.subplots(figsize=(15,6))
    x = np.arange(len(positions)-0.5)
    y = np.abs(X1[positions] - X2[positions])
    bottoms = np.zeros_like(y)
    for i in range(1,4):
        bottoms[:,i] = bottoms[:,i-1] + y[:,i-1]
    ax.set_xticks(x)
    xTickLabels = (1+positions)
    ax.set_xticklabels(xTickLabels, rotation=45)
    ax.set_xlabel('Position', fontsize=textLabelSize)
    ax.set_ylabel('Differences', fontsize=textLabelSize)
    for i in range(4):
        ax.bar(x, y[:,i], color=COLORS[i], label=GATTACA[i], bottom=bottoms[:,i])
    ax.legend(loc='best')
    plt.tight_layout()
