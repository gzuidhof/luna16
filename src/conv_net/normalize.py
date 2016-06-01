from __future__ import division
import numpy as np

#Single value as opposed to mean/std image
#Perhaps we want to change it to one value per color channel
def calc_mean_std(X):
    mean = np.mean(X)
    std = np.std(X)
    return mean, std

def normalize(data, mean, std):
    return (data-mean)/std
