from __future__ import division
import numpy as np
import LoadImages
import glob

from functools import partial
from multiprocessing import Pool
from sklearn.externals import joblib

from scipy.ndimage import binary_closing, binary_dilation, binary_erosion

DATA_PATH = "data/subset0/"
test_images = glob.glob(DATA_PATH + "output/*.mhd")


def calculate_dice(train,truth,filename):
    dice = np.sum(train[truth>0])*2.0 / (np.sum(train) + np.sum(truth))
    #print "Dice {0:.5f}, overlap {1:.5f}\n".format(dice, np.mean(truth==train)),
    if dice < 0.5:
        print "Failure for file", filename, "dice=",dice

    return dice


def process_image(name, do_closing, closing_structure):
    image_train,_,_ = LoadImages.load_itk_image(name)
    name = name.replace("output","truth")
    image_truth,_,_ = LoadImages.load_itk_image(name)
    truth = np.zeros(image_truth.shape, dtype=np.uint8)
    truth[image_truth >0]=1
    if do_closing:
        image_train = binary_closing(image_train, closing_structure,1)

    score = calculate_dice(image_train,truth, name)

    return score

if __name__ == "__main__":
    dice_scores = []

    #Let's multiprocess
    pool = Pool(processes=4)

    #Take subset
    print "N images", len(test_images)


    kernel7 = np.array([[[0,0,1,1,1,0,0],
            [0,1,1,1,1,1,0],
            [1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1],
            [0,1,1,1,1,1,0],
            [0,0,1,1,1,0,0]]])


    kernel9 = np.array([[[0,0,0,1,1,1,0,0,0],
                         [0,1,1,1,1,1,1,1,0],
                         [0,1,1,1,1,1,1,1,0],
                         [1,1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1,1],
                         [0,1,1,1,1,1,1,1,0],
                         [0,1,1,1,1,1,1,1,0],
                         [0,0,0,1,1,1,0,0,0]]])

    kernel = kernel9
    process_func = partial(process_image, do_closing=True, closing_structure=kernel)
    scores = pool.map(process_func, test_images)

    print "\n---"
    print "Kernel size", kernel.shape
    print "mean: ",np.mean(scores)
    print "median", np.median(scores)
    print "standard deviation: ",np.std(scores)

    joblib.dump(scores, 'data/subset0/dice_scores.pkl')
