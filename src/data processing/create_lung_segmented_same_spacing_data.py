import glob
import numpy as np
import os
import SimpleITK as sitk
import skimage.transform
import scipy.ndimage

from joblib import Parallel, delayed

RESIZE_SPACING = [1, 1, 1]
SAVE_FOLDER = '1_1_1mm'

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def save_itk(image, origin, spacing, filename):
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)

def reshape_image(imageDir, subsetDir):
    if os.path.isfile(imageDir.replace('original',SAVE_FOLDER)) == False:
        img, origin, spacing = load_itk(imageDir)
        mask, _, _ = load_itk(imageDir.replace('{}'.format(subsetDir),'data\lung_masks'))
        mask[mask >0] = 1
        img *= mask

        resize_factor = spacing / RESIZE_SPACING
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize = new_shape / img.shape
        new_spacing = spacing / real_resize
                
        img = scipy.ndimage.interpolation.zoom(img, real_resize)

        origin = origin[::-1]
        new_spacing = new_spacing[::-1]
        save_itk(img,origin,new_spacing,imageDir.replace('original',SAVE_FOLDER))

if __name__ == "__main__":
    for subset in range(10):
        subsetDir = 'data\\original\\subset{}'.format(subset)
        imageNames = glob.glob("{}/*.mhd".format(subsetDir))
        Parallel(n_jobs=4)(delayed(reshape_image)(imageDir,subsetDir) for imageDir in imageNames)