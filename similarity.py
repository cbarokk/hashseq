'''
Computes similarity metric between two event streams.

Author: Axel.Tidemann@telenor.com
'''

import argparse
from collections import defaultdict

import pyspike as spk
import redis
import numpy as np

class SpikeTrain:
    def __init__(self, stream):
        self.stream = defaultdict(list)
        self.min = np.inf
        self.max = -np.inf
        for event in stream.split(','):
            timestamp, event_id = event.split('-')
            timestamp = float(timestamp)
            self.stream[event_id].append(timestamp)

            if timestamp < self.min:
                self.min = timestamp
            if timestamp > self.max:
                self.max = timestamp

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
            

def spike_metric(s1, s2):
    '''The event streams must be on the form SECOND_SINCE_THE_EPOCH-EVENT_ID'''
    S1 = SpikeTrain(s1)
    S2 = SpikeTrain(s2)

    return S1.compare(S2)

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
    
    
