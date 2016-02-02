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

class IncrementDict(dict):
    '''Creates an index for the elements in the dict. 
    Only adds new elements, like a set with an index.'''
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
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

        print 'There are {} rows in {}, from {} to {}.'.format(nrows, args.path, start_date, last_date)

        get_data = partial(store.select, args.frame_table, columns=[args.source, args.event], chunksize=args.chunk_size)

        print 'Training data: {} days, from {} to {}'.format(args.train_days, start_date, end_train_date)
        train_data = get_data(where=from_to(start_date, end_train_date))
        aggregate(train_data, train_sources, event_IDs)

        print 'Validation data: {} days, from {} to {}'.format(args.val_days, end_train_date, end_val_date)
        val_data = get_data(where=from_to(end_train_date, end_val_date))
        aggregate(val_data, val_sources, event_IDs)
        
    print 'Found {} train sources, {} val sources, {} events.'.format(len(train_sources), len(val_sources), len(event_IDs))
    return train_sources, val_sources, event_IDs

def trim_data(len_seq, data):
    delete = []
    for s in data:
        if len(data[s]) < len_seq+1:
            delete.append(s)
    for d in delete:
        del data[d]
    print '...kept {} sources'.format(len(data))

def fetch_k_events(k, data):
    s = random.sample(data.keys(), 1)[0]
    if k == -1:
        return ','.join([ str(x) for x in data[s]])

    start = random.randint(0, len(data[s])-k-1) 
    return ','.join([ str(x) for x in data[s][start:start+k]])

def dump_id_mapping(event_IDs):
    f = open('event_IDs_mapping.txt','w')
    for k,v in event_IDs.iteritems():
        f.write(str(k) + " : "+ str(v) + "\n")
    f.close() 

def push(train_sources, val_sources, len_seq, prefix):
    cnt = 0
    fill_val = False
    fill_train = False
    while True:
        if fill_train:
            red.rpush('{}-train'.format(prefix), fetch_k_events(len_seq, train_sources))
        if fill_val:
            red.rpush('{}-validate'.format(prefix), fetch_k_events(len_seq, val_sources))
        cnt = cnt + 1
        if cnt % 1000 == 0:
            cnt = 0
            while True:
                fill_val = False
                fill_train = False
                train_llen = red.llen('{}-train'.format(prefix))
                val_llen = red.llen('{}-validate'.format(prefix))
                if train_llen < 1000:
                    fill_train = True
                    
                if val_llen < 1000:
                    fill_val = True

                if fill_train or fill_val:
                    break
                time.sleep(1.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path',
        help='Path to HDF5 file')
    parser.add_argument(
        '--len_seq',
        help='Length of sequences for each device. -1 means the entire sequence.',
        default=-1)
    parser.add_argument(
        '--train_days',
        help='Number of days to use for training.',
        default=1)
    parser.add_argument(
        '--val_days',
        help='Number of days to use for validation.',
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
        default=5e4)
    parser.add_argument(
        '--queue_prefix',
        help='The prefix for the redis queue',
        default='tacos')
                                                   
    args = parser.parse_args()

    red = redis.Redis("localhost")

    train_sources, val_sources, event_IDs = load_data(args)
    if args.len_seq > 0:
        trim_data(args.len_seq, train_sources)
        trim_data(args.len_seq, val_sources)
    dump_id_mapping(event_IDs)
    print 'Starting to push data to redis'
    push(train_sources, val_sources, args.len_seq, args.queue_prefix)
        
