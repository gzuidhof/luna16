from __future__ import division
import numpy as np
import LoadImages
import glob
import operator

from functools import partial
from multiprocessing import Pool

from scipy.ndimage import binary_closing
from sklearn.externals import joblib

import matplotlib.pyplot as plt

DATA_PATH = "data/subset0/"
test_images = glob.glob(DATA_PATH + "output/*.mhd")

def process_image(name):
    image,_,_ = LoadImages.load_itk_image(name)
    volume = np.sum(image)/np.product(image.shape)

    return volume



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
