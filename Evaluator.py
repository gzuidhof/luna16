import numpy as np
import LoadImages
import glob

DATA_PATH = "data/subset0/subset0"
test_images = glob.glob(DATA_PATH + "mask/*.mhd")



def calculate_dice(train,truth):
    dice = np.sum(train[truth>0])*2.0 / (np.sum(train) + np.sum(truth))
    return dice


if __name__ == "__main__":
    dice_scores =[]
    for name in test_images:
        image_train,_,_ = LoadImages.load_itk_image(name)
        name = name.replace("mask","truth")
        image_truth,_,_ = LoadImages.load_itk_image(name)
        truth = np.zeros(image_truth.shape)
        truth[image_truth >0]=1
        score = calculate_dice(image_train,truth)
        print score
        dice_scores.append(score)
    print "mean: ",np.mean(dice_scores)
    print "standard deviaton: ",np.std(dice_scores)

