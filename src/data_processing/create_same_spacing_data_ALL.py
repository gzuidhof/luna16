import glob
import numpy as np
import os
import SimpleITK as sitk
import skimage.transform
import scipy.ndimage
import pandas as pd
import cPickle as pickle
import gzip
import matplotlib.pyplot as plt
import time

from xyz_utils import load_itk, world_2_voxel, voxel_2_world 
from joblib import Parallel, delayed

RESIZE_SPACING = [1, 1, 1]
SAVE_FOLDER_image = '1_1_1mm_slices_lung_ALL'
SAVE_FOLDER_lung_mask = '1_1_1mm_slices_lung_masks_ALL'
SAVE_FOLDER_nodule_mask = '1_1_1mm_slices_nodule_ALL'


def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    else:
        return([])

def draw_circles(image,cands,origin,spacing):

	#make empty matrix, which will be filled with the mask
    image_mask = np.zeros(image.shape)

    #run over all the nodules in the lungs
    for ca in cands.values:

    	#get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4])/2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_x,coord_y,coord_z))

        #determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord,origin,spacing)


        #determine the range of the nodule
        noduleRange = seq(-radius, radius, RESIZE_SPACING[0])

        #create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_x+x,coord_y+y,coord_z+z)),origin,spacing)
                    if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) < radius:
                        image_mask[np.round(coords[0]),np.round(coords[1]),np.round(coords[2])] = int(1)
    
    return image_mask


def create_slices(imagePath, maskPath, cads):
    #if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
    img, origin, spacing = load_itk(imagePath)
    mask, _, _ = load_itk(maskPath)

    #determine the cads in a lung from csv file
    imageName = os.path.split(imagePath)[1].replace('.mhd','')
    image_cads = cads[cads['seriesuid'] == imageName]

    #calculate resize factor
    resize_factor = spacing / RESIZE_SPACING
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize
    
    #resize image & resize lung-mask
    lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)
    lung_mask = scipy.ndimage.interpolation.zoom(mask, real_resize)
    #print lung_mask.shape()

    #lung_mask to 0-1
    lung_mask[lung_mask >0] = 1
    
    #create nodule mask
    nodule_mask = draw_circles(lung_img,image_cads,origin,new_spacing)

    #Determine which slices contain nodules
    sliceList = []
    for z in range(nodule_mask.shape[2]):
    	sliceList.append(z)

    #save slices
    for z in sliceList:
        lung_slice = lung_img[:,:,z]
        lung_mask_slice = lung_mask[:,:,z]
        nodule_mask_slice = nodule_mask[:,:,z]
        
        #padding to 512x512
        original_shape = lung_img.shape
        lung_slice_512 = np.zeros((512,512)) - 3000
        lung_mask_slice_512 = np.zeros((512,512))
        nodule_mask_slice_512 = np.zeros((512,512))

        offset = (512 - original_shape[1])
        upper_offset = np.round(offset/2)
        lower_offset = offset - upper_offset

        new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)

        lung_slice_512[upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_slice
        lung_mask_slice_512[upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask_slice
        nodule_mask_slice_512[upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask_slice

        #save the lung slice
        savePath = imagePath.replace('original_lungs',SAVE_FOLDER_image)
        file = gzip.open(savePath.replace('.mhd','_slice{}.pkl.gz'.format(z)),'wb')
        pickle.dump(lung_slice_512,file,protocol=-1)
        pickle.dump(new_spacing,file, protocol=-1)
        pickle.dump(new_origin,file, protocol=-1)
        file.close()

        savePath = imagePath.replace('original_lungs',SAVE_FOLDER_lung_mask)
        file = gzip.open(savePath.replace('.mhd','_slice{}.pkl.gz'.format(z)),'wb')
        pickle.dump(lung_mask_slice_512,file,protocol=-1)
        pickle.dump(new_spacing,file, protocol=-1)
        pickle.dump(new_origin,file, protocol=-1)
        file.close()

        savePath = imagePath.replace('original_lungs',SAVE_FOLDER_nodule_mask)
        file = gzip.open(savePath.replace('.mhd','_slice{}.pkl.gz'.format(z)),'wb')
        pickle.dump(nodule_mask_slice_512,file,protocol=-1)
        pickle.dump(new_spacing,file, protocol=-1)
        pickle.dump(new_origin,file, protocol=-1)
        file.close()
		
		# Open File With following code:
        #file = gzip.open(imagePath.replace('.mhd','_slice{}.pkl.gz'.format(z)),'rb')
        #l_slice = pickle.load(file)
        #l_spacing = pickle.load(file)
        #l_origin = pickle.load(file)
        #file.close()

def createImageList(subset, cads):
    imagesWithNodules = []
    subsetDir = 'data\\original_lungs\\subset{}'.format(subset) 	
    imagePaths = glob.glob("{}/*.mhd".format(subsetDir))
    return imagePaths

if __name__ == "__main__":
    cads = pd.read_csv("csv\\annotations.csv")
    for subset in range(10):
    	start_time = time.time()
        print '{} - Processing subset'.format(time.strftime("%H:%M:%S")), subset
        imagePaths = createImageList(subset, cads)
        Parallel(n_jobs=4)(delayed(create_slices)(imagePath,imagePath.replace('original_lungs\\subset{}'.format(subset), 'original_lung_masks'), cads) for imagePath in imagePaths)
        print '{} - Processing subset {} took {} seconds'.format(time.strftime("%H:%M:%S"),subset,np.floor(time.time()-start_time))
        print