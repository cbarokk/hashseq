import sys
from collections import defaultdict
from datetime import datetime
import random
import argparse
import os
import shutil

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity
import redis
import numpy as np
import ipdb

from similarity import spike_metric_matrix, sequence_to_minutes_and_events

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
    if target_distance == 0:
        yield []
        return
    for candidate in haystack:
        distance_so_far = 0
        for ch1, ch2 in zip(needle, candidate):
            distance_so_far += ch1 != ch2
            if distance_so_far > target_distance:
                break
            
        if distance_so_far == target_distance:
            yield candidate

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


    #plt.legend()
    
def plot_similarity(similarity, title, path):
    plt.figure()
    plt.title(title)
    upper_triangle_indices = np.triu_indices(len(similarity),k=1)
    plt.plot(sorted(similarity[upper_triangle_indices]))
    plt.ylim((0,1.01))
    plt.savefig(path, dpi=300)
    plt.clf()

def plot_events(sequences, similarity, title, path):
    plt.figure()
    plt.title(title)
    colors = cm.rainbow(np.linspace(0, 1, len(sequences)))

    for seq, score, color in zip(sequences, similarity[:,0], colors):
        minutes, events = sequence_to_minutes_and_events(seq)
        plt.scatter(minutes, events, color=color, label='score: {0:.2f}'.format(score))

    plt.legend()
    plt.savefig(path, dpi=300)
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--num_keys',
        help='The number of keys to retrieve from redis',
        type=int)
    parser.add_argument(
        '--distance',
        help='The hamming distance from the chosen key',
        type=int,
        default=1)
    parser.add_argument(
        '--plot_similarity',
        help='Plot sorted similarities to a given key.',
        action='store_true')
    parser.add_argument(
        '--plot_events',
        help='Plot events in a week',
        action='store_true')
    parser.add_argument(
        '--figure_folder',
        help='Where to put plotted figures',
        default='hamming_figures')
    parser.add_argument(
        '--random',
        help='Compare streams at random',
        action='store_true')
        
    args = parser.parse_args()

    print 'Deleting folder {}'.format(args.figure_folder)
    shutil.rmtree(args.figure_folder)
    os.makedirs(args.figure_folder)
        
    if args.random:
        while True:
            print "comparing random sequences..."
            sequences = []
            for key in get_random_sequences(10):
                sequences.append(random.choice(list(red.smembers(key))))
            similarity = spike_metric_matrix(sequences)
            if args.plot:
                plot_similarity(similarity,
                                'code length: {} bits'.format(len(key)),
                                '{}/{}.png'.format(args.figure_folder, key))
            print similarity

    for key in get_random_sequences(args.num_keys):
        sequences = list(red.smembers(key))
        all_keys = red.keys("*0*")

        for neighbour in find_hamming_neighbours(key, all_keys, args.distance):
            sequences += list(red.smembers(neighbour))

        if len(sequences) == 1:
            continue

        similarity = spike_metric_matrix(sequences)

        assert np.max(similarity) <= 1.0, 'FIX YOUR SIMILARITY METRIC, FOOL!'

        if args.plot_similarity:
            plot_similarity(similarity,
                            'code length: {} bits'.format(len(key)),
                            '{}/similarity_{}.png'.format(args.figure_folder, key))

        if args.plot_events:
            plot_events(sequences, similarity,
                        'code length: {} bits'.format(len(key)),
                        '{}/events_{}.png'.format(args.figure_folder, key))
