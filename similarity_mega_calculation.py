'''
Plots the similarity between streams in keys.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import random

import redis
import numpy as np

from hamming_ball import find_hamming_neighbours
from similarity import spike_metric_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--num_comparisons',
        help='The number of comparisons to make',
        type=int,
        default=100000)
    parser.add_argument(
        '--distance',
        help='The hamming distance from the chosen key',
        type=int,
        default=1)
    parser.add_argument(
        '--random',
        help='Compare streams at random',
        action='store_true')
    args = parser.parse_args()

    red = redis.StrictRedis()

    n = int(np.sqrt(2*args.num_comparisons + 1./4) - 1./2) + 2
    
    print 'In order to do {} comparisons, we need {} sequences.'.format(args.num_comparisons, n)
    results = []    
    while len(results) < args.num_comparisons:
        all_keys = red.keys("*0*")
        random.shuffle(all_keys)    
        for key in all_keys:
            if args.random:
                sequences = [ random.choice(list(red.smembers(key)))
                              for key in all_keys[:n] ]
            else:
                sequences = list(red.smembers(key))
                for neighbour in find_hamming_neighbours(key, all_keys, args.distance):
                    sequences += list(red.smembers(neighbour))

                    if sum(range(len(sequences))) + len(results) > args.num_comparisons:
                        break
                    
                min_cmp = np.searchsorted(np.cumsum(range(len(sequences))),
                                      args.num_comparisons - len(results),
                                      side='right')
                sequences = sequences[:min_cmp]


            if len(sequences) == 1:
                continue

            similarity = spike_metric_matrix(sequences)
            upper_triangle_indices = np.triu_indices(len(similarity),k=1)
            results += list(similarity[upper_triangle_indices])
            
            print 'len seq: {} -> {} results, total result length: {}'.format(len(sequences), sum(range(len(sequences))), len(results))
            
            if len(results) >= args.num_comparisons:
                break

    print 'Compared {} stream pairs.'.format(len(results))
    results = results[:args.num_comparisons]
    
    txt = 'random' if args.random else 'distance_{}'.format(args.distance)
    np.savetxt('comparisons_{}_{}.txt'.format(args.num_comparisons, txt),
               sorted(results))
