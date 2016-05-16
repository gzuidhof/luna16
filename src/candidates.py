from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
import pandas as pd
from tqdm import tqdm


#Merge candidates of a single scan
def merge_candidates_scan(candidates, distance=5.):
    distances = pdist(candidates, metric='euclidean')
    adjacency_matrix = squareform(distances)

    # Determine nodes within distance, replace by 1 (=adjacency matrix)
    adjacency_matrix = np.where(adjacency_matrix<=distance,1,0)

    # Determine all connected components in the graph
    n, labels = connected_components(adjacency_matrix)

    new_candidates = np.zeros((n,3))

    # Take the mean for these connected components
    for cluster_i in range(n):
        points = candidates[np.where(labels==cluster_i)]
        new_candidates[cluster_i,:] = np.mean(points,axis=0)

    return new_candidates

def merge_candidates(df_candidates, distance=5.):
    new_candidates = np.zeros((0,3))
    for scan_name in tqdm(df_candidates['seriesuid'].unique()):
        #print "Merging scan", scan_name
        df = df_candidates[df_candidates['seriesuid']==scan_name]
        x = df['coordX']
        y = df['coordY']
        z = df['coordZ']
        shape = (len(x),3)
        candidates = np.zeros(shape)
        candidates[:,0]=x
        candidates[:,1]=y
        candidates[:,2]=z

        new = merge_candidates_scan(candidates, distance)
        new_candidates = np.append(new_candidates, new,axis=0)

    print new_candidates


def load_candidates(filename, as_coords=False):
    candidates = pd.read_csv(filename)
    return candidates


# Save candidates given filename and pandas dataframe
# Dataframe with columns:
# seriesuid, coordX, coordY, coordZ, class
# class seems to be 0 always
def save_candidates(filename, df_candidates):
    pass

if __name__ == "__main__":
    df = load_candidates('../data/candidates.csv')
    merge_candidates(df)
