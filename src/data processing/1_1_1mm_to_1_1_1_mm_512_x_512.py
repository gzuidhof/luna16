import glob
import numpy as np
import os
import SimpleITK as sitk
import skimage.transform
import scipy.ndimage

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

def padding_to_512(imagePath):
	if os.path.isfile(imagePath.replace('1_1_1mm_lung_segmentations','1_1_1mm_512_x_512_lung_segmentations')) == False:
		lung_img, origin, spacing = load_itk(imagePath)
		anno_img, _, _ = load_itk(imagePath.replace('lung_segmentations','annotation_masks'))

		original_shape = lung_img.shape
		lung_img_512 = np.zeros((original_shape[0],512,512), dtype=lung_img.dtype)
		anno_img_512 = np.zeros((original_shape[0],512,512), dtype=anno_img.dtype)

		offset = (512 - original_shape[1])
		upper_offset = np.round(offset/2)
		lower_offset = offset - upper_offset

		new_origin = voxel_2_world([0,-upper_offset,-lower_offset],origin,spacing)

		lung_img_512[:,upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_img
		anno_img_512[:,upper_offset:-lower_offset,upper_offset:-lower_offset] = anno_img

		new_origin = new_origin[::-1]
		new_spacing = spacing[::-1]
		save_itk(lung_img_512,new_origin,new_spacing,imagePath.replace('1_1_1mm_lung_segmentations','1_1_1mm_512_x_512_lung_segmentations'))
		save_itk(anno_img_512,new_origin,new_spacing,imagePath.replace('1_1_1mm_lung_segmentations','1_1_1mm_512_x_512_annotation_masks'))

if __name__ == "__main__":
    for subset in range(10):
    	print 'Processing subset', subset
        segLungDir = 'data\\1_1_1mm_lung_segmentations\\subset{}'.format(subset)
        imageNames = glob.glob("{}/*.mhd".format(segLungDir))
        Parallel(n_jobs=4)(delayed(padding_to_512)(imagePath) for imagePath in imageNames)