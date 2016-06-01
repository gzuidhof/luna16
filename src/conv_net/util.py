from __future__ import division
import numpy as np
import pandas as pd
from params import *
import os
from multiprocessing.pool import ThreadPool
import cv2
from visualize import visualize_data

def float32(k):
    return np.cast['float32'](k)

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
        from http://goo.gl/DZNhk
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

# ### Batch iterator ###
# This is just a simple helper function iterating over training
# data in mini-batches of a particular size, optionally in random order.
# It assumes data is available as numpy arrays.
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def histogram_equalization(images, adaptive=True):

    _images = np.array(images * 255, dtype = np.uint8)

    pool = ThreadPool(4)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def process_image(image):
        #print image.shape, image.dtype
        image = image.transpose(1,2,0)

        if adaptive:
            image[:,:,0] = clahe.apply(image[:,:,0])
            image[:,:,1] = clahe.apply(image[:,:,1])
            image[:,:,2] = clahe.apply(image[:,:,2])
        else:
            image[:,:,0] = cv2.equalizeHist(image[:,:,0])
            image[:,:,1] = cv2.equalizeHist(image[:,:,1])
            image[:,:,2] = cv2.equalizeHist(image[:,:,2])

        image = image.transpose(2,0,1)
        return image

    equalized = pool.map(process_image, _images)
    equalized = np.array(equalized, dtype=np.float32)/255.

    #visualize_data(np.append(images[:8],equalized[:8],axis=0).transpose(0,2,3,1))
    return equalized



def hsv_augment(im, hue, saturation, value):
    """
    Augments an image with additive hue, saturation and value.

    `im` should be 01c RGB in range 0-1.
    `hue`, `saturation` and `value` should be scalars between -1 and 1.

    Return value: a 01c RGB image.
    """

    # Convert to HSV
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    # Rescale hue from 0-360 to 0-1.
    im[:, :, 0] /= 360.

    # Mask value == 0
    black_indices = im[:, :, 2] == 0

    # Add random hue, saturation and value
    im[:, :, 0] = (im[:, :, 0] + hue) % 1
    im[:, :, 1] = im[:, :, 1] + saturation
    im[:, :, 2] = im[:, :, 2] + value

    # Pixels that were black stay black
    im[black_indices, 2] = 0

    # Clip pixels from 0 to 1
    im = np.clip(im, 0, 1)

    # Rescale hue from 0-1 to 0-360.
    im[:, :, 0] *= 360.

    # Convert back to RGB in 0-1 range.
    im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)


    return im
