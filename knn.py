import redis
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import sys

red = redis.Redis("localhost")
#red = redis.Redis(host="193.156.17.90")

def read_vectors(filename, domain):
   f = open(filename, 'r')
   vectors = {}
   mapping = {}
   pipe = red.pipeline()
   cnt=0
   for line in f:
      words = line.split()
      vectors[words[0]] = np.array(words[1:]).astype(np.float)
      #mapping[cnt] = words[0]
      mapping[cnt] = words[0]
      cnt +=1
      pipe.sadd(domain +":vocab", words[0])

   pipe.execute()
   return vectors, mapping

def find_nearest_neighbors(mapping, D, domain):
   pipe = red.pipeline()

   for i in range(len(mapping)):
      tmp={}
      neighbors = [x[0] for x in enumerate(D[i,:]) if x[1] <= 0.5]
      for j in range(len(neighbors)):
         if not(mapping[i] ==  mapping[neighbors[j]]):
            neighbor = mapping[neighbors[j]]
            #pipe.zadd(domain + ":"+mapping[i], neighbor=D[i,neighbors[j]])
            tmp[mapping[neighbors[j]]] = D[i,neighbors[j]]
            #print D[i,neighbors[j]], " : ", mapping[i], " -->" , mapping[neighbors[j]]
            if len(tmp):
               pipe.zadd(domain + ":"+mapping[i], **tmp)

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

   print "loading vectors"
   vectors, mapping = read_vectors(filename, domain)

   X = []
   for i in range(len(mapping)):
      X.append(vectors[mapping[i]])
   
   X = np.array(X)

   print "computing distances"
   D = pairwise_distances(X, metric='cosine', n_jobs=1)

   print "storing in redis"
   find_nearest_neighbors(mapping, D, domain)

   #indices = get_indexes(["norway", "ceo", "malaysia"], mapping)
   #Y = np.add( np.subtract(X[indices[0],:], X[indices[1],:]), X[indices[2],:] ) 
   #Y = np.reshape(Y, (1, 50))
   #print Y.shape
   #D2 = pairwise_distances(X, Y, metric='cosine', n_jobs=-1)
   #print mapping[np.argmin(D2)]




