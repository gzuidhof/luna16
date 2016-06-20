import gzip
import cPickle as pickle
import numpy as np

from params import params as P

def load_images (image_paths):
    X = []

    for image_path in image_paths
        with gzip.open(image_path):
        X.append(pickle.load(file))

    X = np.array(X)
    
    if P.ZERO_CENTER:
        X -= P.MEAN_PIXEL

    return X