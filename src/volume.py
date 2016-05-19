from __future__ import division
import numpy as np
import image_read_write
import glob
import operator
from skimage.draw import circle
from functools import partial
from multiprocessing import Pool

from scipy.ndimage import binary_closing
from sklearn.externals import joblib

import matplotlib.pyplot as plt

DATA_PATH = "data/subset1/"
test_images = glob.glob(DATA_PATH + "subset1mask/*.mhd")
threshold = -350

def process_image(name):
    image,_,_ = image_read_write.load_itk_image(name)
    volume = np.sum(image)/np.product(image.shape)

    return volume

def process_failure(name):
    name = name.replace("mask","truth")
    name2 = name.replace("truth","")
    image,_,_ = image_read_write.load_itk_image(name2)
    #image_cropped = image[:,120:420,60:460]
    image_mask = np.zeros(image.shape)
    center = 256
    cc,rr = circle(center+20,center,160)
    image_mask[:,cc,rr] = 1
    image[image>threshold]=0
    image[image!=0]=1
    image = image*image_mask
    #image_cropped[image_cropped>threshold]=0
    #image_cropped[image_cropped!=0]=1

    kernel20 = np.zeros((15,15))
    cc,rr = circle(7,7,8)
    kernel20[cc,rr]=1
    image = binary_closing(image, [kernel20],1)
    #image[:,:,:]=0
    #image[:,120:420,60:460]=image_cropped
    truth,_,_ = image_read_write.load_itk_image(name)
    print evaluator.calculate_dice(image,truth,name)
    image = np.array(image,dtype=np.int8)
    #LoadImages.save_itk(image,name)

if __name__ == "__main__":

    #Let's multiprocess
    pool = Pool(processes=8)

    #Take subset
    #test_images = test_image[:20]

    print "N images", len(test_images)

    volumes = pool.map(process_image, test_images)

    scores = joblib.load(DATA_PATH+'dice_scores.pkl')

    zp = zip(volumes, scores, test_images)
    zp.sort(key=operator.itemgetter(1))


    #Determine failed segmentations
    def is_failed(tup):
        vol, score, name = tup
        return vol < 0.02
    failures = filter(is_failed, zp)

    print "Sorted by score"
    for x in zp:
        print x

    print "Failures:"
    for failure in failures:
        print failure
        process_failure(failure[2])
