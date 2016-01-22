import redis
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import sys

red = redis.Redis("localhost")
#red = redis.Redis(host="193.156.17.90")

def read_mapping(filename):
    f = open(filename, 'r')
    mapping = {}
    
    for line in f:
        words = line.split(":")
        mapping[int(words[1])] = int(words[0])
    return mapping

def read_vectors(filename, domain, mapping):
   f = open(filename, 'r')
   vectors = {}
   pipe = red.pipeline()
   for line in f:
      words = line.split()
      key = int(words[0].split(":")[0])
      if key in mapping:
         vectors[key] = np.array([float(x) for x in words[1:]]).astype(np.float) 
         pipe.sadd(domain +":vocab", mapping[key])

   pipe.execute()
   return vectors

def find_nearest_neighbors(mapping, D, domain):
   pipe = red.pipeline()

   #for i in range(len(mapping)):
   for i in mapping.keys():
      tmp={}
      neighbors = [x[0] for x in enumerate(D[i-1,:]) if x[1] <= 0.5]
      for j in range(len(neighbors)):
         if not(i-1 ==  neighbors[j]):
            tmp[mapping[neighbors[j]+1]] = D[i-1,neighbors[j]]
            print D[i-1,neighbors[j]], " : ", mapping[i], " -->" , mapping[neighbors[j]+1]
      if len(tmp):
         pipe.hmset(domain + ":" + str(mapping[i]), tmp)
      pipe.execute()


def get_indexes(words, mapping):
   tmp = {}
   for k, v in mapping.items():
      if v in words:
         tmp[v] = k

   res = []
   for w in words:
      res.append(tmp[w])
   return res

if __name__ == '__main__':

   filename = sys.argv[1]

   domain = "globul:" + sys.argv[2]

   map_file = sys.argv[3]
   mapping = read_mapping(map_file)    

   print "loading vectors"
   vectors = read_vectors(filename, domain, mapping)

   X=[]
   for i in sorted(vectors):
      X.append(vectors[i])
   X = np.array(X)

   print "computing distances"
   D = pairwise_distances(X, metric='cosine', n_jobs=1)

   print "storing in redis"
   find_nearest_neighbors(mapping, D, domain)

