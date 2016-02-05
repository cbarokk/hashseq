'''
Loads TACOS data into redis database.

Author: Cyril.Banino-Rokkones@telenor.com, Axel.Tidemann@telenor.com
'''

import argparse
from collections import defaultdict
import sys
import random
from datetime import datetime
import time
import json
from functools import partial

import pandas as pd
import numpy as np
import redis
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

class IncrementDict(dict):
    '''Creates an index for the elements in the dict. 
    Only adds new elements, like a set with an index.'''
    def __init__(self):
        dict.__init__(self)
        self.counter = 1

    def add(self, key):
        if key not in self:
            self[key] = self.counter
            self.counter += 1
            
def aggregate(data, aggregator, event_IDs):
    for chunk in data:
        for row in chunk.itertuples():
            timestamp, source, event = row
            event_IDs.add(event)
            aggregator[source].append('{}-{}'.format(timestamp.strftime('%s'), event_IDs[event]))

    for name in aggregator.keys():
        aggregator[name] = sorted(aggregator[name])

def from_to(start_date, end_date):
    return 'index > "{}" & index < "{}"'.format(start_date, end_date)

def load_data(args):
    train_sources = defaultdict(list)
    val_sources = defaultdict(list)
    event_IDs = IncrementDict()

    with pd.get_store(args.path) as store:
        nrows = store.get_storer(args.frame_table).nrows

        start_date = store.select(args.frame_table, start=0, stop=1).index[0]
        last_date = store.select(args.frame_table, start=nrows-1, stop=nrows).index[0]

        end_train_date = start_date + pd.Timedelta('{} days'.format(args.train_days))
        end_val_date = end_train_date + pd.Timedelta('{} days'.format(args.val_days))

        print 'There are {} rows in {}, from {} to {}. Assuming file is sorted.'.format(nrows, args.path, start_date, last_date)

        get_data = partial(store.select, args.frame_table, columns=[args.source, args.event], chunksize=args.chunk_size)

        print 'Training data: {} days, from {} to {}'.format(args.train_days, start_date, end_train_date)
        train_data = get_data(where=from_to(start_date, end_train_date))
        aggregate(train_data, train_sources, event_IDs)

        print 'Validation data: {} days, from {} to {}'.format(args.val_days, end_train_date, end_val_date)
        val_data = get_data(where=from_to(end_train_date, end_val_date))
        aggregate(val_data, val_sources, event_IDs)
        
    print 'Found {} train sources, {} val sources, {} unique events.'.format(len(train_sources), len(val_sources), len(event_IDs))
    return train_sources, val_sources, event_IDs

def trim_data(args, data, filename=False):
    if filename:
        histogram(data, filename)

    for s in data.keys():
        if not args.lower_len_seq < len(data[s]) < args.upper_len_seq:
            del data[s]

    if filename:
        histogram(data, '{} trimmed'.format(filename))
    print 'Trimming sequence lengths to ({},{}), keeping {} sources.'.format(args.lower_len_seq, args.upper_len_seq, len(data))

def fecth_k_events(k, data):
    x = random.choice(data.values())
    start = random.randint(0, len(x)-k)
    return ','.join(x[start:start+k])

def dump_id_mapping(event_IDs):
    with open('event_IDs_mapping.txt','w') as f:
        for k,v in event_IDs.iteritems():
            f.write('{} : {}\n'.format(str(k), str(v)))

def histogram(data, filename):
    sns.distplot([ len(x) for x in data.values() ], kde=False)
    plt.title(filename)
    plt.tight_layout()
    plt.savefig('{}.png'.format(filename), dpi=300)
    plt.clf()
            
def push(train_sources, val_sources, event_IDs, prefix, k):
    red = redis.StrictRedis()

    train_queue = '{}-train'.format(prefix)
    val_queue = '{}-validate'.format(prefix)

    red.set('{}-num_events', len(event_IDs))
    
    while True:
        if red.llen(train_queue) < 1000:
            red.rpush(train_queue, fetch_k_events(train_sources))
        if red.llen(val_queue) < 1000:
            red.rpush(val_queue, fetch_k_events(val_sources))

        if red.llen(train_queue) > 750 and red.llen(val_queue) > 750:
            time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path',
        help='Path to HDF5 file')
    parser.add_argument(
        '--upper_len_seq',
        help='Upper bound sequence length for each device.',
        type=int,
        default=np.inf)
    parser.add_argument(
        '--lower_len_seq',
        help='Lower bound sequence length for each device.',
        type=int,
        default=1)
    parser.add_argument(
        '--train_days',
        help='Number of days to use for training.',
        type=int,
        default=1)
    parser.add_argument(
        '--val_days',
        help='Number of days to use for validation.',
        type=int,
        default=1)
    parser.add_argument(
        '--frame_table',
        help='The name of the frame table in the HDF5 file',
        default='tacos')
    parser.add_argument(
        '--source',
        help='The column that holds the sources of the data',
        default='node')
    parser.add_argument(
        '--event',
        help='The column that holds the events in the data',
        default='alarmtype')
    parser.add_argument(
        '--chunk_size',
        help='Chunk size to iterate over HDF5 file',
        type=int,
        default=50000)
    parser.add_argument(
        '--queue_prefix',
        help='The prefix of the redis queue, for train/validation queues',
        default='tacos')
    args = parser.parse_args()

    train_sources, val_sources, event_IDs = load_data(args)
    trim_data(args, train_sources)
    trim_data(args, val_sources)
    dump_id_mapping(event_IDs)
    print 'Starting to push data to redis'
    push(train_sources, val_sources, event_IDs, args.queue_prefix, args.lower_len_seq+1)
