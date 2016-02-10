import redis
import numpy as np
import sys
from collections import defaultdict
from datetime import datetime
import random
import argparse

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from similarity import spike_metric_matrix

red = redis.StrictRedis()

def load_embedings(filename):
    f = open(filename, 'r')
    vectors = {}
    
    for line in f:
        words = line.split()
        vectors[int(words[0].split(":")[0])] = [float(x) for x in words[1:]] 
      
    return vectors


event_embed = load_embedings("embed_event.txt")
theta_embed = load_embedings("embed_theta.txt")


def split(sequence):
    return [ map(int, events.split('-')) for events in sequence.split(',') ]

def get_key_members(key, sequences):
    members = red.smembers(key)
    for m in members:
        seq = []
        events = m.split(",")
        for e in events:
            words = e.split("-")
            seq.append(words)
        sequences.append(seq)

def compute_distances(sequences):
    std = []
    for i in range(len(sequences[0])):
        times = []
        for seq in sequences:
            times.append(int(seq[i][0]))
        std.append(np.std(times))
    return (np.average(std), std[-1])

def get_random_sequences(num):
    codes = red.keys("*0*")
    return random.sample(codes, num)

def find_hamming_neighbours(needle, haystack, target_distance):
    neighbours = []
    for candidate in haystack:
        distance_so_far = 0
        for ch1, ch2 in zip(needle, candidate):
            distance_so_far += ch1 != ch2
            if distance_so_far > target_distance:
                break
        if distance_so_far == target_distance:
            neighbours.append(candidate)
    return neighbours

def hamming_distance(s1, s2):
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

def plot_live(distances, num_streams):
    pass
    #plt.title(key)
    #plt.xlim((0, 50))
    #plt.ylim((0, 51))

    #plt.scatter(distances[0], distances[1], s=30
    # plt.scatter(distances[0], distances[1], s=30
    # plt.draw()
    #plt.pause(0.05)
    #plt.pause(3)

    # for i, d in enumerate(streams):
    #     lines[i][0].remove()
    #     del lines[i][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'num_keys',
        help='The number of keys to retrieve from redis',
        type=int)
    parser.add_argument(
        '--distance',
        help='The hamming distance from the chosen key',
        type=int,
        default=1)
    parser.add_argument(
        '--plot',
        help='Whether to plot time and event IDs of similar streams',
        type=bool,
        default=False)
    args = parser.parse_args()

    plt.ion()

    if args.num_keys == 0:
        while True:
            print "comparing random sequences..."
            sequences = []
            for key in get_random_sequences(10):
                sequences += list(red.smembers(key))
            print spike_metric_matrix(sequences)

    for key in get_random_sequences(args.num_keys):
        sequences = list(red.smembers(key))
        all_keys = red.keys("*0*")
        for neighbour in find_hamming_neighbours(key, all_keys, args.distance):
            sequences += list(red.smembers(neighbour))

        if len(sequences) == 1:
            continue

        if args.plot:
            plt.figure()
            
        for s in sequences:
            print '{} ... {}'.format(s[:95], s[-95:])
            if args.plot:
                x, y = zip(*split(s))
                plt.plot(x,y)
                plt.draw()
        
        print spike_metric_matrix(sequences)

        # distances = compute_distances(sequences)                       
        # plot_live(distances, len(seqences))



    
            
    
