import SimpleITK as sitk
import numpy as np
import normalize as norm
from skimage import filter as filters
from joblib import Parallel
from skimage import exposure
from skimage import feature
import matplotlib.pyplot as plt
from tqdm import tqdm

def return_surrounding(coords, image_array, radius):
    patch = image_array[coords[0], coords[1]-radius:coords[1]+radius, coords[2]-radius:coords[2]+radius]

    return patch

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpy_image = sitk.GetArrayFromImage(itkimage)

    numpy_origin = np.array(list(reversed(itkimage.GetOrigin())))
    numpy_spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpy_image, numpy_origin, numpy_spacing

def show_images(images):
    for image in images:
        plt.figure()
        plt.imshow(image, cmap='Greys_r')
    plt.show()

def threshold_by_histogram(image):
    val = filters.threshold_otsu(image)
    return image < val

def label_image(image):
    l = 4
    n = 20
    image = filters.gaussian(image, sigma=l / (4. * n))
    blobs = image > image.mean()
    return blobs

def blob_image_multiscale2(image, type=0):
    # function that return a list of blob_coordinates, 0 = dog, 1 = doh, 2 = log
    list = []
    image = norm.normalize(image)
    for z, slice in tqdm(enumerate(image)):
        # init list of different sigma/zoom blobs
        featureblobs = []
        # x = 0,1,2,3,4
        for x in xrange(5):
            if type == 0:
                featureblobs[x] = feature.blob_dog(slice, 2^x, 2^(x+1))
            if type == 1:
                featureblobs[x] = feature.blob_doh(slice, 2^x, 2^(x+1))
            if type == 2:
                featureblobs[x] = feature.blob_log(slice, 2^x, 2^(x+1))
        # init list of blob coords
        blob_coords = np.zeros((len(featureblobs[0]),4))
        # start at biggest blob size
        for featureblob in reversed(featureblobs):
            i = 0
            # for every blob found of a blobsize
            for blob in enumerate(featureblob):
                # if that blob is not within range of another blob, add it
                if(not within_range(blob, blob_coords)):
                    blob_coords[i] = [z, blob[0], blob[1], blob[2]]
                    i = i + 1
        list.append(blob_coords[0:3])
    return list

def blob_image_multiscale3(image, type=0):
    # function that return a list of blob_coordinates, 0 = dog, 1 = doh, 2 = log
    list = []
    image = norm.normalize(image)
    for z, slice in tqdm(enumerate(image)):
        # init list of different sigma/zoom blobs
        featureblobs = []
        # x = 0,1,2
        for x in xrange(3):
            if type == 0:
                featureblobs[x] = feature.blob_dog(slice, 3^x, 3^(x+1))
            if type == 1:
                featureblobs[x] = feature.blob_doh(slice, 3^x, 3^(x+1))
            if type == 2:
                featureblobs[x] = feature.blob_log(slice, 3^x, 3^(x+1))
        # init list of blob coords
        blob_coords = np.zeros((len(featureblobs[0]),4))
        # start at biggest blob size
        for featureblob in reversed(featureblobs):
            i = 0
            # for every blob found of a blobsize
            for blob in enumerate(featureblob):
                # if that blob is not within range of another blob, add it
                if not within_range(blob, blob_coords):
                    blob_coords[i] = [z, blob[0], blob[1], blob[2]]
                    i = i + 1
        list.append(blob_coords[0:3])
    return list

def within_range(blob, blob_coords):
    for coords in blob_coords:
        if((blob[0] - coords[1])^2 + (blob[1] - coords[2]) < coords[3]^2):
            return 1
    return 0

def blob_image(image):
    #img_path    =   '../1.3.6.1.4.1.14519.5.2.1.6279.6001.100332161840553388986847034053.mhd'
    # slice = numpy_image[240,:,:]
    # normalized = norm.normalize(return_surrounding([240,240,240],numpy_image, 240))
    # thresholded = threshold_by_histogram(normalized)
    # blobs = label_image(thresholded)
    # show_images([blobs, normalized, thresholded])

    # normalized3d = norm.normalize(numpy_image)
    # thresholded3d = threshold_by_histogram(normalized3d)
    list = []
    image = norm.normalize(image)
    #print "normalized and thresholded"

    for z, slice in tqdm(enumerate(image)):
        blobs = feature.blob_doh(slice)
        #print blobs.shape
        blob_coords = np.zeros((len(blobs),3))
        for i, blob in enumerate(blobs):
            blob_coords[i] = [z, blob[0], blob[1]]
        list.append(blob_coords)

    #print list
    return list
