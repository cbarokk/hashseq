'''
Computes similarity metric between two event streams.

Author: Axel.Tidemann@telenor.com
'''

import argparse
from collections import defaultdict
from datetime import datetime

import pyspike as spk
import redis
import numpy as np

class SpikeTrain:
    def __init__(self, stream):
        self.stream = defaultdict(list)

        minutes, events = sequence_to_minutes_and_events(stream)

        self.min = min(minutes)
        self.max = max(minutes)

        for minute, event in zip(minutes, events):
            self.stream[event].append(minute)

        for event in self.stream.keys():
            self.stream[event] = sorted(self.stream[event])


    def compare(self, other):
        first = min([self.min, other.min])
        last = max([self.max, other.max])

        edges = (first, last)
        
        common_keys = set(self.stream.keys()).intersection(other.stream.keys())
        union_keys = set(self.stream.keys()).union(other.stream.keys())
        
        simi = [ spk.spike_sync(spk.SpikeTrain(self.stream[key], edges),
                                spk.SpikeTrain(other.stream[key], edges))
                 for key in common_keys ]

        return sum(simi)/len(union_keys)

def sequence_to_minutes_and_events(sequence):
    seconds, events = zip(*split(sequence))
    minutes = map(minute_of_the_week, seconds)
    return minutes, events
    
def split(sequence):
    return [ map(int, events.split('-')) for events in sequence.split(',') ]
    
def minute_of_the_week(timestamp):
    date = datetime.fromtimestamp(timestamp)
    weekday = int(date.weekday())
    hour = int(date.strftime('%H'))
    minute = int(date.strftime('%M'))
    weekly_bin = (weekday*1440 + hour*60 + minute)

    return weekly_bin
    
def spike_metric(s1, s2):
    '''The event streams must be on the form SECOND_SINCE_THE_EPOCH-EVENT_ID. 
    They are wrapped into weekly intervals.'''
    S1 = SpikeTrain(s1)
    S2 = SpikeTrain(s2)

    return S1.compare(S2)

def spike_metric_matrix(sequences):
    metric = np.zeros((len(sequences), len(sequences)))
    for i, s1 in enumerate(sequences):
        for j, s2 in enumerate(sequences):
            if i < j: # Symmetrical
                metric[i,j] = spike_metric(s1, s2)
    return metric
                      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--redis_queue',
        help='Which redis queue to read examples from',
        default='tacos-train')
    args = parser.parse_args()

    red = redis.StrictRedis()
    _, event_stream1 = red.blpop(args.redis_queue, 0)
    _, event_stream2 = red.blpop(args.redis_queue, 0)

    print 'Comparing \n {} \n to \n {}'.format(event_stream1, event_stream2)
    print spike_metric(event_stream1, event_stream2)
    
    
