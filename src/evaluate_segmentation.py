from __future__ import division
import numpy as np
import image_read_write
import glob

from functools import partial
from multiprocessing import Pool
from sklearn.externals import joblib

from scipy.ndimage import binary_closing, binary_dilation, binary_erosion

DATA_PATH = "data/subset1/"
test_images = glob.glob(DATA_PATH + "subset1mask/*.mhd")


def calculate_dice(train,truth,filename):
    dice = np.sum(train[truth>0])*2.0 / (np.sum(train) + np.sum(truth))
    #print "Dice {0:.5f}, overlap {1:.5f}\n".format(dice, np.mean(truth==train)),
    if dice < 0.5:
        print "Failure for file", filename, "dice=",dice

    return dice


def process_image(name, do_closing, closing_structure):
    image_train,_,_ = image_read_write.load_itk_image(name)
    name = name.replace("mask","truth")
    image_truth,_,_ = image_read_write.load_itk_image(name)
    truth = np.zeros(image_truth.shape, dtype=np.uint8)
    truth[image_truth >0]=1
    if do_closing:
        image_train = binary_closing(image_train, closing_structure,1)

    image_train = binary_closing(image_train, [[[1]],[[1]],[[1]],[[1]],[[1]]],1)

    score = calculate_dice(image_train,truth, name)

    return score

def determine_dice_scores():
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

    kernel12 = np.array([[[0,0,0,0,1,1,1,1,0,0,0,0],
                     [0,0,1,1,1,1,1,1,1,1,0,0],
                     [0,1,1,1,1,1,1,1,1,1,1,0],
                     [0,1,1,1,1,1,1,1,1,1,1,0],
                     [1,1,1,1,1,1,1,1,1,1,1,1],
                     [1,1,1,1,1,1,1,1,1,1,1,1],
                     [1,1,1,1,1,1,1,1,1,1,1,1],
                     [1,1,1,1,1,1,1,1,1,1,1,1],
                     [0,1,1,1,1,1,1,1,1,1,1,0],
                     [0,1,1,1,1,1,1,1,1,1,1,0],
                     [0,0,1,1,1,1,1,1,1,1,0,0],
                     [0,0,0,0,1,1,1,1,0,0,0,0]]])

    kernel = kernel9
    process_func = partial(process_image, do_closing=True, closing_structure=kernel)
    scores = pool.map(process_func, test_images)

    print "\n---"
    print "Kernel size", kernel.shape
    print "mean: ",np.mean(scores)
    print "median", np.median(scores)
    print "standard deviation: ",np.std(scores)

    joblib.dump(scores, 'data/subset1/dice_scores.pkl')

def normalize_slices_of_image(filename):
    im = image_read_write.load_itk_image_rescaled(filename,slice_mm=1.0)
    image_read_write.save_itk(im, filename.replace('output','rescaled'))
    print im.shape

def normalize_slices():
    pool = Pool(processes=3)
    pool.map(normalize_slices_of_image, test_images)
    print "Done!"

import SimpleITK as sitk
if __name__ == "__main__":
    #normalize_slices()

    determine_dice_scores()
    # images = []
    # for name in test_images:
    #     print name
    #     #try:
    #     im = LoadImages.load_itk_image_rescaled(name,1)
    #
    #     LoadImages.save_itk(im, name.replace('output','rescaled'))
    #     #images.append(im)
    #     print im.shape
    #     #except:
    #         #print "Failed!"
