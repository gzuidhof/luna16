
import Image
import augment
from skimage import util
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
import numpy as np


if __name__ == "__main__":
    image = imread("example.jpg")
    #print image
    image = image[:,:,1]
    # plt.subplot(211)
    # image=np.array(image)
    # print image.shape
    # plt.imshow(list(image))
    # image2  = augment.augment([image])
    # plt.subplot(212)
    # plt.imshow(image2[0])
    # print image2[0].shape
    # plt.show()
    images = augment.testtime_augmentation(np.array(image))
    print len(images)
    plt.subplot(221)
    plt.imshow(images[0])
    plt.subplot(222)
    plt.imshow(images[1])
    plt.subplot(223)
    plt.imshow(images[2])
    plt.subplot(224)
    plt.imshow(images[3])
    plt.show()
