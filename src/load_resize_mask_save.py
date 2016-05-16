import glob
import numpy as np
import SimpleITK as sitk
import skimage.transform
import scipy.ndimage

RESIZE_SPACING = [1, 1, 1]

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


counter = 0

if __name__ == "__main__":
    for subset in range(10):
        subsetDir = 'subset{}'.format(subset)
        imageNames = glob.glob("{}/*.mhd".format(subsetDir))
        for imageDir in imageNames:

            if counter == 0:
                counter = 1

                print imageDir

                img, origin, spacing = load_itk(imageDir)
                mask, _, _ = load_itk(imageDir.replace('{}'.format(subsetDir),'lung_masks'))
                print 'Mask:', mask.shape, 'original:', img.shape

                mask[mask >0] = 1
                img *= mask
                print img.shape

                #print 'Spacing:',spacing
                resize_factor = spacing / RESIZE_SPACING
                #print 'Resize_Factor:',resize_factor
                #print 'Old shape:',img.shape
                new_real_shape = img.shape * resize_factor
                #print 'New real shape:', new_real_shape
                new_shape = np.round(new_real_shape)
                print 'new shape:',new_shape
                real_resize = new_shape / img.shape
                #print 'real resize:', real_resize
                new_spacing = spacing / real_resize
                print 'New spacing:',new_spacing

                print 
                #save_itk(img,'original.mhd')
                #img = skimage.transform.resize(img, new_shape)
                #img = skimage.transform.rescale(img,0.1)
                
                img = scipy.ndimage.interpolation.zoom(img, real_resize)

                origin = origin[::-1]
                new_spacing = new_spacing[::-1]

                save_itk(img,origin,new_spacing,'hoi.mhd')