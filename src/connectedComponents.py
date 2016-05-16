import SimpleITK as sitk
import numpy as np
import normalize as norm
from skimage import filters
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt

def show_surrounding(coords, image_array, radius):
    patch = image_array[coords[0], coords[1]-radius:coords[1]+radius, coords[2]-radius:coords[2]+radius]

    return patch

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpy_image = sitk.GetArrayFromImage(itkimage)

    numpy_origin = np.array(list(reversed(itkimage.GetOrigin())))
    numpy_spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpy_image, numpy_origin, numpy_spacing

img_path    =   '1.3.6.1.4.1.14519.5.2.1.6279.6001.100332161840553388986847034053.mhd'

numpy_image, numpy_origin, numpy_spacing = load_itk_image(img_path)
slice = numpy_image[0,:,:]
val = filters.threshold_otsu(slice)
mask = slice < val
plt.imshow(norm.normalize(show_surrounding([240,240,240],numpy_image, 240)), cmap='Greys_r')
plt.show()


