import SimpleITK as sitk
import numpy as np
import normalize as norm
from skimage import filter as filters
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
        #print blobs
        blob_coords = np.zeros((len(blobs),3))
        for i, blob in enumerate(blobs):
            blob_coords[i] = [z, blob[0], blob[1]]
        list.append(blob_coords)

    #print list
    return list
