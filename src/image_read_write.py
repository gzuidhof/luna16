from __future__ import division
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
import pandas


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def load_itk_image_rescaled(filename, slice_mm):
    im, origin, spacing = load_itk_image(filename)

    new_im = zoom(im, [spacing[0]/slice_mm,1.0,1.0])
    return new_im

def save_itk(image, filename):
    im = sitk.GetImageFromArray(image, isVector=False)
    sitk.WriteImage(im, filename, True)

