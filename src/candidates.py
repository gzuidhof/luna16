from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
import pandas as pd

def merge_candidates(candidates, distance=20.):
    distances = pdist(candidates, metric='euclidean')
    adjacency_matrix = squareform(distances)

    #Determine nodes within distance, replace by 1 (=adjacency matrix)
    adjacency_matrix = np.where(adjacency_matrix<=distance,1,0)

    #Determine all connected components in the graph
    n, labels = connected_components(adjacency_matrix)

    new_candidates = np.zeros((n,3))

    #Take the mean for these connected components
    for cluster_i in range(n):
        points = candidates[np.where(labels==cluster_i)]
        new_candidates[cluster_i,:] = np.mean(points,axis=0)

    return new_candidates

def load_candidates(filename, as_coords=False):
    candidates = pd.read_csv(filename)
    return candidates




if __name__ == "__main__":
    df = load_candidates('../data/candidates.csv')
    df = df[df['seriesuid']=='1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860']
    x = df['coordX']
    y = df['coordY']
    z = df['coordZ']

    shape = (len(x),3)
    candidates = np.zeros(shape)

    candidates[:,0]=x
    candidates[:,1]=y
    candidates[:,2]=z

    #print candidates[:10]
    merge_candidates(candidates)
