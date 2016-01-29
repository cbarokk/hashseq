'''
Finds beacons that are similar. Like word embeddings.

Author: Cyril.Banino-Rokkones@telenor.com, Axel.Tidemann@telenor.com
'''

import argparse
from collections import defaultdict, namedtuple
import sys
import random
from datetime import datetime
import time
import json
import multiprocessing as mp

import pandas as pd
import numpy as np
import redis
import psycopg2
from split import chop

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--len_seq',
    help="Length of sequences",
    default=51)
parser.add_argument(
    '--ratio',
    help="Train/validate ratio",
    default=3./4)
args = parser.parse_args()

red = redis.Redis("localhost")

class Data:
    def __init__(self, raw):
        try:
            json_data = json.loads(raw)
            self.entity_id = json_data['entity_id']
            self.timestamp = datetime.strptime(json_data['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')
            self.categories = json_data['categories']
            self.enduser_id = json_data['enduser_id']
            self.OK = json_data['dwell'] == 0 and self.entity_id is not None # We only care about entering the beacon, not exit time
        except:
            self.OK = False

    def to_string(self, beacon_ID):
        return '{}-{}'.format(self.timestamp.strftime('%s'), beacon_ID)

def parse(raw_data):
    result = []
    for row in raw_data:
        _, raw_data = row
        data = Data(raw_data)
        if data.OK:
            result.append(data)
    return result
    

def load_data(collapse=False):
    train_sources = defaultdict(list)
    val_sources = defaultdict(list)
    beacon_IDs = {}

    conn = psycopg2.connect("dbname='tgf2016' user='tgf2016' host='tgf2016.clzkl0f5olbq.eu-west-1.rds.amazonaws.com' password='voubahMowai3ku'")
    cur = conn.cursor()
    cur.execute("SELECT * FROM events")

    raw_result = cur.fetchall()
    print '{} data points in total (we will use half of that since we only look at dwell entry times, not exit).'.format(len(raw_result))
    pool = mp.Pool()
    result = [ item for sublist in pool.map(parse, chop(mp.cpu_count(), raw_result)) for item in sublist ]
    pool.close()
    
    next_id = 1 # 1-indexing in Lua
    for i, data in enumerate(result):
        source = train_sources if i < len(result)*args.ratio else val_sources

        prev_beacon_ID = int(source[data.enduser_id][-1].split('-')[-1]) if data.enduser_id in source else False # Ugliest line today.

        if collapse and (data.entity_id in beacon_IDs) and (prev_beacon_ID == beacon_IDs[data.entity_id]):
            continue

        if not data.entity_id in beacon_IDs:        
            beacon_IDs[data.entity_id] = next_id
            next_id += 1

        string = data.to_string(beacon_IDs[data.entity_id])
        source[data.enduser_id].append(string)
        
    # To make sure it is in order
    for name in train_sources.keys():
        train_sources[name] = sorted(train_sources[name])
    for name in val_sources.keys():
        val_sources[name] = sorted(val_sources[name])

    print 'Found {} train sources, {} val sources, {} beacons.'.format(len(train_sources), len(val_sources), len(beacon_IDs))

    return train_sources, val_sources, beacon_IDs

def trim_data(len_seq, data):
    delete = []
    for s in data:
        if len(data[s]) < len_seq+1:
            delete.append(s)
    for d in delete:
        del data[d]
    print "kept ", len(data), " sources"

def fetch_k_events(k, data):
    s = random.sample(data.keys(), 1)[0]
    start = random.randint(0, len(data[s])-k-1)
    return ",".join([ str(x) for x in data[s][start:start+k]])

def dump_id_mapping(beacon_IDs):
    f = open('beacon_IDs_mapping.txt','w')
    for k,v in beacon_IDs.iteritems():
        f.write(str(k) + " : "+ str(v) + "\n")
    f.close() 

def push(train_sources, val_sources, len_seq):
    cnt = 0
    fill_val = False
    fill_train = False
    while True:
        if fill_train:
            red.rpush('train', fetch_k_events(len_seq, train_sources))
        if fill_val:
            red.rpush('validate', fetch_k_events(len_seq, val_sources))
        cnt = cnt + 1
        if cnt % 1000 == 0:
            cnt = 0
            while True:
                fill_val = False
                fill_train = False
                train_llen = red.llen('train')
                val_llen = red.llen('validate')
                if train_llen < 1000:
                    fill_train = True
                    
                if val_llen < 1000:
                    fill_val = True

                if fill_train or fill_val:
                    break
                time.sleep(1.0)
                #print ("sleep ", train_llen, val_llen)

if __name__ == '__main__':
    train_sources, val_sources, beacon_IDs = load_data(collapse=True)
    print 'train_sources distribution'
    print [ len(name) for name in train_sources.values() ]
    print 'val_sources distribution'
    print [ len(name) for name in val_sources.values() ]
    trim_data(args.len_seq, train_sources)
    trim_data(args.len_seq, val_sources)
    dump_id_mapping(beacon_IDs)
    print 'Starting to push data to redis'
    push(train_sources, val_sources, args.len_seq)
        
