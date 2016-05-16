import SimpleITK as sitk
import numpy as np
import normalize as norm
from IPython.core.display import Image, display
import matplotlib.pyplot as plt

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
        plt.imshow(image)
    plt.show()


img_path    =   '1.3.6.1.4.1.14519.5.2.1.6279.6001.100332161840553388986847034053.mhd'
numpy_image, numpy_origin, numpy_spacing = load_itk_image(img_path)
slice = numpy_image[240,:,:]
normalized = norm.normalize(return_surrounding([240,240,240],numpy_image, 240))
show_images([slice,normalized])


