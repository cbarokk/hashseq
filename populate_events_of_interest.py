'''
Puts the given events in the list for interesting events, or selects a few at random 
from the existing list of events.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import random

import redis

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--path',
    help='Path to text file where there is one element of interest per line',
    default=False)
parser.add_argument(
    '--random',
    help='Randomly select N events to put in the queue of interest',
    type=int,
    default=0)
parser.add_argument(
    '--prefix',
    help='The prefix of the *-events redis list name',
    default='tacos')

args = parser.parse_args()

red = redis.StrictRedis()

events = red.hgetall('{}-events'.format(args.prefix))

if args.random:
    sample = random.sample(events.items(), args.random)
    print 'Putting {} in {}-events-of-interest'.format(sample, args.prefix)
    event_IDs = [ value for _,value in sample ]
else:
    with open(args.path) as f:
        event_IDs = [ events[line.strip()] for line in f ]

red.rpush('{}-events-of-interest'.format(args.prefix), *event_IDs)
