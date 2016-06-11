from __future__ import division
import os.path
import numpy as np
from unet import INPUT_SIZE, OUTPUT_SIZE
import normalize
import gzip
import cPickle as pickle
import loss_weighting
import skimage.morphology
from augment import augment

from params import params as P

_EPSILON = 1e-8

def get_image(filename, deterministic):
    with gzip.open(filename,'rb') as f:
        lung = pickle.load(f)

    truth_filename = filename.replace('lung','nodule')
    segmentation_filename = filename.replace('lung','lung_mask')

    if os.path.isfile(truth_filename):
        with gzip.open(filename.replace('lung','nodule'),'rb') as f:
            truth = np.array(pickle.load(f),dtype=np.float32)
    else:
        truth = np.zeros_like(lung)

    if os.path.isfile(segmentation_filename):
        with gzip.open(segmentation_filename,'rb') as f:
            outside = np.where(pickle.load(f)>0,1.0,0.0)
    else:
        outside = np.where(lung==0,1.0,0.0)

    if P.ERODE_SEGMENTATION > 0:
        kernel = skimage.morphology.disk(P.ERODE_SEGMENTATION)
        outside = skimage.morphology.binary_erosion(outside, kernel)

    if P.AUGMENT and not deterministic:
        lung, truth, outside = augment([lung,truth, outside])

    truth = np.array(np.round(truth),dtype=np.int64)
    outside = np.array(np.round(outside),dtype=np.int8)

    #Set label of outside pixels to -10
    truth = truth - outside

    lung = crop_or_pad(lung, INPUT_SIZE, -400)
    truth = crop_or_pad(truth, OUTPUT_SIZE, 0)
    outside = crop_or_pad(outside, OUTPUT_SIZE, 0)

    lung = normalize.normalize(lung)
    lung = np.expand_dims(np.expand_dims(lung, axis=0),axis=0)

    if P.ZERO_CENTER:
        lung = lung - P.MEAN_PIXEL

    truth = np.array(np.expand_dims(np.expand_dims(truth, axis=0),axis=0),dtype=np.int64)

    return lung, truth

def crop_or_pad(image, desired_size, pad_value):
    if image.shape[0] < desired_size:
        return np.pad(lung, (desired-size-image.shape[0])//2, 'constant', constant_values=-pad_value)
    else:
        offset = (image.shape[0]-desired_size)//2
        return image[offset:offset+desired_size,offset:offset+desired_size]

def load_images(filenames, deterministic=False):
    slices = [get_image(filename, deterministic) for filename in filenames]
    lungs, truths = zip(*slices)

    l = np.array(np.concatenate(lungs,axis=0), dtype=np.float32)
    t = np.concatenate(truths,axis=0)

    # Weight the loss by class balancing, classes other than 0 and 1
    # get set to 0 (the background is -1)
    w = loss_weighting.weight_by_class_balance(t, classes=[0,1])

    #Set -10 labels back to label 0
    t = np.clip(t, 0, 100000)

    return l, t, w, filenames
