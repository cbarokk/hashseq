'''
Looks at saved .t7 files in a folder in order to find the one with the lowest loss, for 
each saved model configuration.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import glob

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--lower_epoch_bound',
    help='Lower epoch bound to search for (exclusive)',
    type=int,
    default=1)
parser.add_argument(
    '--upper_epoch_bound',
    help='Upper epoch bound to search for (exclusive)',
    type=int,
    default=np.inf)
parser.add_argument(
    '--folder',
    help='Which folder to look in',
    default='cv')
args = parser.parse_args()

# We know that all model names are defined before the epoch starts
files = glob.glob('{}/*epoch1*'.format(args.folder))

unique_models = np.unique([ name[:name.find('epoch')] for name in files ])

for model in unique_models:
    index, loss, filename = [], [], []
    for candidate in glob.glob('{}*'.format(model)):
        _, numbers = candidate.split('epoch')
        index.append(int(numbers.split('.')[0]))
        loss.append(float(numbers.split('_')[-1].split('.t7')[0]))
        filename.append(candidate)

    data = pd.DataFrame(index=index, data=zip(loss, filename), columns=['loss', 'filename'])
    data.index.name = 'epoch'

    min_epoch = data.query('index > {} & index < {}'.format(args.lower_epoch_bound, args.upper_epoch_bound)).loss.argmin()

    winner = data.query('index == {}'.format(min_epoch))
    print winner.filename.values[0]
