from __future__ import division
import numpy as np
import LoadImages
import glob

DATA_PATH = "data/subset0/"
test_images = glob.glob(DATA_PATH + "mask/*.mhd")


def calculate_dice(train,truth):
    dice = np.sum(train[truth>0])*2.0 / (np.sum(train) + np.sum(truth))
    print "Dice {0:.5f}, overlap {1:.5f}\n".format(dice, np.mean(truth==train)),
    return dice


if __name__ == "__main__":
    dice_scores =[]

    for name in test_images:
        image_train,_,_ = LoadImages.load_itk_image(name)
        name = name.replace("mask","truth")
        image_truth,_,_ = LoadImages.load_itk_image(name)
        truth = np.zeros(image_truth.shape, dtype=np.uint8)
        truth[image_truth >0]=1
        #print name
        score = calculate_dice(image_train,truth)
        dice_scores.append(score)

    print "\n---\nmean: ",np.mean(dice_scores)
    print "standard deviation: ",np.std(dice_scores)
