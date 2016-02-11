'''
Plots the similarity between streams in keys.

Author: Axel.Tidemann@telenor.com
'''

import os
import argparse
import random
import glob

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from hamming_ball import find_hamming_neighbours
from similarity import spike_metric_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--prefix',
        help='Prefix of the files with values',
        default='comparisons_1000000_')
    parser.add_argument(
        '--fig_name',
        help='Figure file name',
        default='comparisons.png')
    
    args = parser.parse_args()

    if os.path.isfile(args.fig_name):
        os.unlink(args.fig_name)
    
    plt.figure()
    for filename in sorted(glob.glob('{}*'.format(args.prefix))):
        values = np.loadtxt(filename)
        if 'distance' in filename:
            label = filename[filename.find('distance'):].replace('_',' ').replace('.txt','')
        else:
            label = 'random'
        plt.plot(values, label=label)
    plt.xlabel('Pairwise comparisons')
    plt.ylabel('Similarity')
    
    ticks = ['0', '200K','400K', '600K','800K','1000K']
    plt.xticks(np.linspace(0,1e6,6), ticks)
    plt.legend(loc=2)
    plt.savefig(args.fig_name, dpi=300)
