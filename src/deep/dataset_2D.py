import gzip
import cPickle as pickle
import numpy as np
from normalize import normalize

from params import params as P

def load_images (image_paths):
    X = []

    labels = []

    for image_path in image_paths:
        with gzip.open(image_path) as file:
        	X.append(pickle.load(file)[0])

        label = [0,1] if "True" in image_path else [1,0]
        labels.append(label)

    X = np.array(X)

    X = normalize(X)

    if P.ZERO_CENTER:
        X -= P.MEAN_PIXEL

    return np.array(X, dtype=np.float32), np.array(labels)
