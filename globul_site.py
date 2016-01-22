import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import redis
import random
from datetime import datetime
import time 

red = redis.Redis("localhost")

path = sys.argv[1]
len_seq = int(sys.argv[2])
train_days = sys.argv[3]

store = pd.get_store(path)
nrows = store.get_storer('data').nrows
print "nrows", nrows
train_sources = defaultdict(list)
val_sources = defaultdict(list)

cell_IDs = {}
next_id = 1

def load_data():
    start_date = store.select('data', start=1, stop=2).index[0]
    end_date = start_date + pd.Timedelta(train_days)
    print "loading data between ", start_date, " and ", end_date
    #where = 'index>"'+str(start_date) + '" & index<"' + str(end_date) + '" & typeCode == "01"'
    where = 'index>"'+str(start_date) + '" & index<"' + str(end_date) + '"' 
    chunk =  store.select('data', columns=['callingSubscriberIMSI', 'cell_ID'], where=where).groupby(['callingSubscriberIMSI'])

    global next_id
    for name, group in chunk:
        for t, e in zip (group.index, group.cell_ID):
            e=int(e)/10
            print "e", e
            if not e in cell_IDs:
                cell_IDs[e] = next_id
                next_id += 1
            train_sources[name].append(t.strftime("%s")+"-"+ str(cell_IDs[e]))
        train_sources[name] = sorted(train_sources[name])
    print "found ", len(train_sources), " train_sources"
    print "found ", len(cell_IDs), " cell_IDs"

    #start_date = end_date
    #end_date = start_date + pd.Timedelta(train_days)
    #end_date = start_date + pd.Timedelta(train_days)
    #print "loading data between ", start_date, " and ", end_date
    #where = 'index>"'+str(start_date) + '" & index<"' + str(end_date) + '"'

    where = 'index>"' + str(end_date) + '"' 
    chunk =  store.select('data', columns=['callingSubscriberIMSI', 'cell_ID'], where=where).groupby(['callingSubscriberIMSI'])
    for name, group in chunk:
        for t, e in zip (group.index, group.cell_ID):
            if not e in cell_IDs:
                print "found cell not in train set...", e
                cell_IDs[e] = next_id
                next_id += 1
            val_sources[name].append(t.strftime("%s")+"-"+ str(cell_IDs[e]))
        val_sources[name] = sorted(val_sources[name])
    print "found ", len(val_sources), " val_sources"

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

def dump_id_mapping():
    f = open('CellIds_mapping.txt','w')
    for k,v in cell_IDs.iteritems():
        f.write(str(k) + " : "+ str(v) + "\n")
    f.close() 

def push():
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
                print ("sleep ", train_llen, val_llen)





if __name__ == '__main__':
    load_data()
    trim_data(len_seq, train_sources)
    trim_data(len_seq, val_sources)
    dump_id_mapping()
    push()
        
