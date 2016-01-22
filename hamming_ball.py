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


def get_key_members(key, sequences):
    members = r.smembers(key)
    for m in members:
        seq = []
        events = m.split(",")
        for e in events:
            words = e.split("-")
            seq.append(words)
        sequences.append(seq)

def compute_distances(seqences):
    std = []
    for i in range(len(sequences[0])):
        times = []
        for seq in sequences:
            times.append(int(seq[i][0]))
        std.append(np.std(times))
    return (np.average(std), std[-1])


def get_random_sequences(num):
    codes = r.keys("*0*")
    return random.sample(codes, num)


def plot_live(distances, num_streams):
    #plt.title(key)
    #plt.xlim((0, 50))
    #plt.ylim((0, 51))

    #plt.scatter(distances[0], distances[1], s=30
    plt.scatter(distances[0], distances[1], s=30
    plt.draw()
    #plt.pause(0.05)
    #plt.pause(3)

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

        for i,b in enumerate(key):
            if b == '0':
                b_ = '1'
            else:
                b_ = '0'
            neighbor = key[:i] + b_ + key[i+1:]
            if r.scard(neighbor):
                print "found " + neighbor
                get_key_members(neighbor, sequences)
        
        if len(sequences) == 1:
            print "did not find neighbors within hamming distance 1..."
            continue

        distances = compute_distances(sequences)                       
        plot_live(distances, len(seqences))



    
            
    
