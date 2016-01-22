import numpy as np
import matplotlib.pyplot as plt
import math
from tsne import bh_sne
from collections import defaultdict
import sys

def read_vectors(filename):
    f = open(filename, 'r')
    vectors = {}
    
    for i, line in enumerate(f):
        words = line.split()
        vectors[i] = [float(x) for x in words] 
      
    return vectors


if __name__ == '__main__':

    filename = sys.argv[1]
    
    vectors = read_vectors(filename)
    
    for v in vectors.items():
        print v
    asd=asf

    X=[]
    #colors = np.array(range(len(vectors)))/float(len(vectors))
    #colors = range(len(vectors))

    for i in sorted(vectors):
        X.append(vectors[i])
    X = np.array(X)

    #TSNE
    #cm = plt.cm.get_cmap('RdYlGn')
    #cm = plt.cm.get_cmap('binary')
    #cm = plt.cm.get_cmap('hsv')
    #cm = plt.cm.get_cmap('rainbow')
    Y = bh_sne(X, perplexity=50)
    #plt.imshow(colors)
    #plt.scatter(Y[:,0], Y[:,1], c=colors, vmin=0, vmax=15, s=50, cmap=cm, linewidth='0')
    #plt.scatter(Y[:,0], Y[:,1], c=colors, s=50, cmap=cm, linewidth='0')
    #plt.scatter(Y[:,0], Y[:,1], s=50, cmap=cm, linewidth='0')
    plt.scatter(Y[:,0], Y[:,1], s=30)
    #plt.colorbar()
    plt.show()

    
    
    #model = TSNE(n_components=2, random_state=0, perplexity=50, early_exaggeration=10, metric="cosine", verbose=2)
    #X_tsne = model.fit_transform(X)
    #plt.scatter(X_tsne[:,0], X_tsne[:,1], 10);
    #plt.show()
	
