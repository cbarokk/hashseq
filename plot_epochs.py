'''
Plots epochs based on saved configuration files.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import glob
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--folder',
    help='Which folder to look in',
    default='cv')

args = parser.parse_args()
plt.figure()

while True:
    plt.clf()
    # We know that all model names are defined before the epoch starts
    files = glob.glob('{}/*epoch1*'.format(args.folder))

    unique_models = np.unique([ name[:name.find('epoch')] for name in files ])
    colors = cm.rainbow(np.linspace(0, 1, len(unique_models)))


    for model, color in zip(unique_models, colors):
        index, loss = [], []
        for candidate in glob.glob('{}*'.format(model)):
            _, numbers = candidate.split('epoch')
            index.append(int(numbers.split('.')[0]))
            loss.append(float(numbers.split('_')[-1].split('.t7')[0]))

        data = pd.DataFrame(index=index, data=loss, columns=['loss'])
        data.index.name = 'epoch'
        data.sort_index(inplace=True)

        arch, layers, size, time_weight, event_weight = model[model.find('model_past_'):].replace('model_past_','').split('_')[:-1]

        plt.plot(data, label='{} {}'.format(arch, size), color=color)
        plt.xlabel('Epochs')
        plt.ylabel('Training loss')

    plt.legend()
    plt.savefig('epochs.png', dpi=300)

    time.sleep(5)
