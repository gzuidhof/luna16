from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
import pandas as pd
from tqdm import tqdm

CANDIDATES_COLUMNS = ['seriesuid','coordX','coordY','coordZ','class']


#Merge candidates of a single scan
def merge_candidates_scan(candidates, seriesuid, distance=5.):
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
        center = np.mean(points,axis=0)
        new_candidates[cluster_i,:] = center

    x = new_candidates[:,0]
    y = new_candidates[:,1]
    z = new_candidates[:,2]
    labels = [seriesuid]*len(x)
    class_name = [0]*len(x)

    data= zip(labels,x,y,z,class_name)

    new_candidates = pd.DataFrame(data,columns=CANDIDATES_COLUMNS)

    return new_candidates

def merge_candidates(df_candidates, distance=5.):
    new_candidates = pd.DataFrame()
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

        new = merge_candidates_scan(candidates,seriesuid=scan_name,distance=distance)
        new_candidates = new_candidates.append(new)


    #print new_candidates
    return new_candidates


def load_candidates(filename, as_coords=False):
    candidates = pd.read_csv(filename)
    return candidates

# Save candidates given filename and pandas dataframe
# Dataframe with columns:
# seriesuid, coordX, coordY, coordZ, class
# class seems to be 0 always
def save_candidates(filename, df_candidates):
    df_candidates.to_csv(filename, index=False)

def coords_to_candidates(coords, seriesuid):
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    names = [seriesuid]*len(x)
    class_name = [0]*len(x)

    data = zip(names,x,y,z,class_name)
    candidates = pd.DataFrame(data,columns=CANDIDATES_COLUMNS)
    return candidates

if __name__ == "__main__":

    #df = load_candidates('../data/candidates.csv')
    #new_candidates = merge_candidates(df)
    #save_candidates('test.csv', new_candidates)
