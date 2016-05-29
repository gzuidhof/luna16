from __future__ import division
import os.path
import numpy as np
from unet import INPUT_SIZE, OUTPUT_SIZE
import normalize
import gzip
#import cPickle as pickle
import pickle

_EPSILON = 1e-8

def get_image(filename):
    #print "----------------"+str(filename)+"----------------"
    with gzip.open(filename,'rb') as f:
        lung = pickle.load(f)

    with gzip.open(filename.replace('lung','nodule'),'rb') as f:
        truth = pickle.load(f)

    lung = np.pad(lung, (INPUT_SIZE-lung.shape[0])//2, 'constant', constant_values=-400)

    lung = np.array(normalize.normalize(lung),dtype=np.float32)

    # Crop truth
    crop_size = OUTPUT_SIZE
    offset = (truth.shape[0]-crop_size)//2
    truth = truth[offset:offset+crop_size,offset:offset+crop_size]

    lung = np.expand_dims(np.expand_dims(lung, axis=0),axis=0)
    lung = lung-0.66200809792889126

    truth = np.array(np.expand_dims(np.expand_dims(truth, axis=0),axis=0),dtype=np.int64)

    true_case_weight = (1/(np.mean(truth)+_EPSILON)) * 1.1
    weights = np.array((true_case_weight-1)*truth + 1, dtype=np.float32)

    return lung, truth, weights

def load_images(filenames):
    filenames = filter(lambda x: x!='.', filenames)
    slices = map(get_image, filenames)
    lungs, truths, weights = zip(*slices)
    return np.concatenate(lungs,axis=0), np.concatenate(truths,axis=0), np.concatenate(weights,axis=0)
