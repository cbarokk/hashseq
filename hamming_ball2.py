import redis
import numpy as np
import sys
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from  sklearn.metrics.pairwise import cosine_similarity
import random

r = redis.Redis("localhost")

def load_embedings(filename):
    f = open(filename, 'r')
    vectors = {}
    
    for line in f:
        words = line.split()
        vectors[int(words[0].split(":")[0])] = [float(x) for x in words[1:]] 
      
    return vectors


event_embed = load_embedings("embed_event.txt")
theta_embed = load_embedings("embed_theta.txt")


def minute_of_the_week(timestamp):
    date = datetime.fromtimestamp(timestamp)
    weekday = int(date.weekday())
    hour = int(date.strftime('%H'))
    minute = int(date.strftime('%M'))
    weekly_bin = (weekday*1440 + hour*60 + minute)

    return weekly_bin



def get_key_members(key, sequences):
    members = r.smembers(key)
    for m in members:
        seq = []
        events = m.split(",")
        for e in events:
            words = e.split("-")
            seq.append(words)
        sequences.append(seq)

def compare_sequences(s1, s2):
    distances = []
    cosine = []
    for i in range(len(s1)):
        t1= minute_of_the_week(int(s1[i][0]))
        t2= minute_of_the_week(int(s2[i][0]))
        distances.append((min(abs(t2-t1), 10080-abs(t2-t1)))/60.0)

        e_1 = np.array(event_embed[int(s1[i][1])]).reshape(1,-1)
        e_2 = np.array(event_embed[int(s2[i][1])]).reshape(1,-1)
        D = cosine_similarity(e_1, e_2)
        cosine.append(D[0][0])
    return distances, cosine

def compute_distances(seqences):
    distances = []
    cosine = []
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            d, c = compare_sequences(sequences[i], sequences[j])
            distances.append(d)
            cosine.append(c)
    return distances, cosine


def get_minutes(seqences):
    minutes = np.empty([len(sequences), len(sequences[0])])
    
    for i, seq in enumerate(sequences):
        for j, e in enumerate(seq):
            minutes[i][j] = minute_of_the_week(int(e[0]))
    return minutes


def get_random_sequences(num):
    codes = r.keys("*0*")
    return random.sample(codes, num)


def plot_live(streams, key):
    plt.title(key)
    plt.xlim((0, 10081))
    plt.ylim((0, 51))
    labels = [
        'Mon',
        '06:00',
        '12:00',
        '18:00',
        'Tue',
        '06:00',
        '12:00',
        '18:00',
        'Wed',
        '06:00',
        '12:00',
        '18:00',
        'Thu',
        '06:00',
        '12:00',
        '18:00',
        'Fri',
        '06:00',
        '12:00',
        '18:00',
        'Sat',
        '06:00',
        '12:00',
        '18:00',
        'Sun',
        '06:00',
        '12:00',
        '18:00',
        'Mon'
    ]
    plt.xticks(range(0, 10081, 360), labels)

    Y = list(reversed(range(len(streams[0]))))
    lines = {}
    for i, d in enumerate(streams):
        lines[i] = plt.plot([],[])
        #lines[i] = plt.scatter([],[])

    for i in range(len(Y)):
        for j, d in enumerate(streams):
            lines[j][0].set_ydata(Y[:i+1])
            lines[j][0].set_xdata(d[:i+1])
        plt.draw()
        plt.pause(0.05)
    plt.pause(3)

    for i, d in enumerate(streams):
        lines[i][0].remove()
        del lines[i][0]

if __name__ == '__main__':
    num_keys = int(sys.argv[1])
    plt.ion()

    if num_keys == 0:
        while True:
            print "comparing random sequences..."
            sequences = []
            for key in get_random_sequences(10):
                get_key_members(key, sequences)
            minutes = get_minutes(sequences)                       
            plot_live(minutes, "RANDOM STREAMS")

    for key in get_random_sequences(num_keys):
        print "searching for neighbors of ", key
        sequences = []
        get_key_members(key, sequences)
        if len(sequences) == 1:
            continue

        #for i,b in enumerate(key):
        #    if b == '0':
        #        b_ = '1'
        #    else:
        #        b_ = '0'
        #    neighbor = key[:i] + b_ + key[i+1:]
        #    if r.scard(neighbor):
        #        print "found " + neighbor
        #        get_key_members(neighbor, sequences)
        #
        #if len(sequences) == 1:
        #    print "did not find neighbors within hamming distance 1..."
        #    continue

        minutes = get_minutes(sequences)                       
        plot_live(minutes, key)



    
            
    
