import glob
import numpy as np
import os
import SimpleITK as sitk
import skimage.transform
import scipy.ndimage
import cPickle as pickle
import gzip

from joblib import Parallel, delayed

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

def world_2_voxel(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def voxel_2_world(voxelCoord, origin, spacing):
    stretchedVoxelCoord = voxelCoord * spacing
    worldCoord = stretchedVoxelCoord + origin
    return worldCoord

def save_slices(imagePath):
    lung_img, _, _ = load_itk(imagePath)
    anno_img, _, _ = load_itk(imagePath.replace('lung_segmentations','annotation_masks'))

    for z in range(lung_img.shape[0]):
        anno_slice = anno_img[z,:,:]
        if np.mean(anno_slice) > 0:
            lung_slice = lung_img[z,:,:]

            lung_path = imagePath.replace('lung_segmentations','lung_slices')
            file = gzip.open(lung_path.replace('.mhd','_slice{}.pkl.gz'.format(z)),'wb')
            pickle.dump(lung_slice,file,protocol=-1)
            file.close()

            # Open File With following code:
            #file = gzip.open(lung_path.replace('.mhd','_slice{}.pkl.gz'.format(z)),'rb')
            #lung_slice2 = pickle.load(file)
            #file.close()

            #Comparison test
            #if np.array_equal(lung_slice,lung_slice2):
            #    print 'Succes!'
            #else:
            #    print 'Fail!'

            nodule_path = imagePath.replace('lung_segmentations','nodule_slices')
            file = gzip.open(nodule_path.replace('.mhd','_slice{}.pkl.gz'.format(z)),'wb')
            pickle.dump(anno_slice,file,protocol=-1)
            file.close()

if __name__ == "__main__":
    for subset in range(10):
        print 'Processing subset', subset
        segLungDir = 'data\\1_1_1mm_512_x_512_lung_segmentations\\subset{}'.format(subset)
        imageNames = glob.glob("{}/*.mhd".format(segLungDir))
        Parallel(n_jobs=4)(delayed(save_slices)(imagePath) for imagePath in imageNames)