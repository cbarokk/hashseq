import numpy as np
import matplotlib.pyplot as plt
import math
from tsne import bh_sne
from collections import defaultdict
import sys

def read_vectors(filename, mapping):
    f = open(filename, 'r')
    vectors = {}
    
    for line in f:
        words = line.split()
        key = int(words[0].split(":")[0])
        if key in mapping:
            vectors[key] = [float(x) for x in words[1:]] 
      
    return vectors

def read_mapping(filename):
    f = open(filename, 'r')
    mapping = {}
    
    for line in f:
        words = line.split(":")
        mapping[int(words[1])] = int(words[0])
    return mapping


if __name__ == '__main__':

    vec_file = sys.argv[1]
    map_file = sys.argv[2]

    mapping = read_mapping(map_file)    
    vectors = read_vectors(vec_file, mapping)

    
    X=[]
    for i in sorted(vectors):
        X.append(vectors[i])
    X = np.array(X)

    colors = np.array(range(len(vectors)))/float(len(vectors))
    colors = range(len(vectors))

    #TSNE
    #cm = plt.cm.get_cmap('RdYlGn')
    #cm = plt.cm.get_cmap('binary')
    #cm = plt.cm.get_cmap('hsv')
    #cm = plt.cm.get_cmap('rainbow')
    Y = bh_sne(X, perplexity=50)
    #plt.imshow(colors)
    #plt.scatter(Y[:,0], Y[:,1], c=colors, vmin=0, vmax=15, s=50, cmap=cm, linewidth='0')
    #plt.scatter(Y[:,0], Y[:,1], c=colors, s=50, cmap=cm, linewidth='0')
    plt.scatter(Y[:,0], Y[:,1], s=50)
    #plt.scatter(Y[:,0], Y[:,1], s=30)
    #plt.colorbar()

    #labels = [mapping[i] for i in range(1, len(vectors)+1)]
    #print "labels", labels
    #for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    #    plt.annotate(
    #        label, 
    #        xy = (x, y), xytext = (-2, 2),
    #        textcoords = 'offset points', ha = 'right', va = 'bottom')
    #        #textcoords = 'offset points', ha = 'right', va = 'bottom',
    #        #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    #        #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    plt.show()

    
    
    #model = TSNE(n_components=2, random_state=0, perplexity=50, early_exaggeration=10, metric="cosine", verbose=2)
    #X_tsne = model.fit_transform(X)
    #plt.scatter(X_tsne[:,0], X_tsne[:,1], 10);
    #plt.show()
	
