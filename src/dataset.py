from __future__ import division
import os.path
import numpy as np
from unet import INPUT_SIZE, OUTPUT_SIZE
import normalize
import gzip
import cPickle as pickle
import loss_weighting
import skimage.morphology

import matplotlib.pyplot as plt

_EPSILON = 1e-8

def get_image(filename):
    with gzip.open(filename,'rb') as f:
        lung = pickle.load(f)

    with gzip.open(filename.replace('lung','nodule'),'rb') as f:
        truth = pickle.load(f)

    #Random transpose
    #if np.random.randint(2)==0:
    #    lung = lung.transpose(1,0)
    #    truth = truth.transpose(1,0)
    #    lung[:,:] = lung[::-1,:]
    #    truth[:,:] = truth[::-1,:]
    #    lung = lung.transpose(1,0)
    #    truth = truth.transpose(1,0)


    #We do not care about the outside
    outside = np.where(lung==0,True,False)
    kernel = skimage.morphology.disk(3)
    outside = skimage.morphology.binary_erosion(outside, kernel)

    #Set label of outside pixels to -1
    truth = truth - outside

    lung = np.pad(lung, (INPUT_SIZE-lung.shape[0])//2, 'constant', constant_values=-400)
    lung = np.array(normalize.normalize(lung),dtype=np.float32)

    # Crop truth
    crop_size = OUTPUT_SIZE
    offset = (truth.shape[0]-crop_size)//2
    truth = truth[offset:offset+crop_size,offset:offset+crop_size]

    lung = np.expand_dims(np.expand_dims(lung, axis=0),axis=0)
    lung = lung-0.66200809792889126

    truth = np.array(np.expand_dims(np.expand_dims(truth, axis=0),axis=0),dtype=np.int64)

    return lung, truth

def load_images(filenames):
    filenames = filter(lambda x: x!='.', filenames)
    slices = map(get_image, filenames)
    lungs, truths = zip(*slices)

    l = np.concatenate(lungs,axis=0)
    t = np.concatenate(truths,axis=0)

    # Weight the loss by class balancing, classes other than 0 and 1
    # get set to 0 (the background is -1)
    w = loss_weighting.weight_by_class_balance(t, classes=[0,1])

    #Set -1 labels back to label 0
    t = np.clip(t, 0, 100000)

    return l, t, w
