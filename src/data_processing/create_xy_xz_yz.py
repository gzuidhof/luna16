import glob
import numpy as np
import scipy.ndimage
import pandas as pd

import cPickle as pickle
import gzip
import os
import time

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from xyz_utils import load_itk, world_2_voxel, voxel_2_world

# OUTPUT_SPACING in mm
OUTPUT_SPACING = [0.5, 0.5, 0.5]
# Output image with have shape [3,OUTPUT_DIM,OUTPUT_DIM]
OUTPUT_DIM = 64
# CANDIDATES_CSV
CANDIDATES_CSV = 'V2'

def process_image(image_path, candidates, save_dir):
    #load image 
    image, origin, spacing = load_itk(image_path)

    #calculate resize factor
    resize_factor = spacing / OUTPUT_SPACING
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / image.shape
    new_spacing = spacing / real_resize
    
    #resize image
    image = scipy.ndimage.interpolation.zoom(image, real_resize)

    #Pad image with offset (OUTPUT_DIM/2) to prevent problems with candidates on edge of image
    offset = OUTPUT_DIM/2
    image = np.pad(image,offset,'constant',constant_values=0)

    #Make a indixlist of the candidates of the image 
    image_name = os.path.split(image_path)[1].replace('.mhd','')
    indices = candidates[candidates['image_name'] == image_name].index

    #loop through the candidates within this image
    for i in indices:

        #get row data and nodule voxel coords
        row = candidates.iloc[i]
        world_coords = np.array([row.x, row.y, row.z])
        # add offset to voxel coords to cope with padding
        coords = np.floor(world_2_voxel(world_coords,origin,new_spacing)) + offset
        label = row.label

        # Create xy, xz, yz
        xy_slice = np.transpose(image[coords[0]-offset:coords[0]+offset,coords[1]-offset:coords[1]+offset,coords[2]])
        xz_slice = np.rot90(image[coords[0]-offset:coords[0]+offset,coords[1],coords[2]-offset:coords[2]+offset])
        yz_slice = np.rot90(image[coords[0],coords[1]-offset:coords[1]+offset,coords[2]-offset:coords[2]+offset])

        # UNCOMMENT THESE LINES IF YOU WANT TO MANUALLY COMPARE IMAGES WITH MEVISLAB
        # test_coords means coords you need to look up in mevislab
        #test_coords = world_2_voxel(world_coords,origin,spacing)
        #print 'x:',test_coords[0],'y:',test_coords[1],'z:',test_coords[2]
        #plt.imshow(xy_slice)
        #plt.gray()
        #plt.figure(2)
        #plt.imshow(xz_slice)
        #plt.gray()
        #plt.figure(3)
        #plt.imshow(yz_slice)
        #plt.gray()
        #plt.show()

        # Create output
        output = np.zeros([3,OUTPUT_DIM,OUTPUT_DIM])
        output[0,:,:] = xy_slice
        output[1,:,:] = xz_slice
        output[2,:,:] = yz_slice

        # Determine save_path based on label and indices (need +2 due to starting with zero + header in csv)
        if label:
        	save_path = save_dir.format('True') + '/{}.pkl.gz'.format(i+2)
        else:
        	save_path = save_dir.format('False') + '/{}.pkl.gz'.format(i+2)

        # save with gzip/pickle
        with gzip.open(save_path,'wb') as f:
        	pickle.dump(output,f,protocol=-1)

        # UNCOMMENT IF YOU WANT TO TEST WHETHER SAVING WORKED
        #with gzip.open(save_path,'rb') as f:
        #    test = pickle.load(f)
        #check = True
        #if np.array_equal(test[0,:,:],xy_slice) == False:
        #    check = False
        #if np.array_equal(test[1,:,:],xz_slice) == False:
        #    check = False
        #if np.array_equal(test[2,:,:],yz_slice) == False:
        #    check = False
        #print check

if __name__ == "__main__":

    candidates = pd.read_csv('csv/candidates_{}.csv'.format(CANDIDATES_CSV))

    for subset in range(7,10):
        start_time = time.time()

        # Prepare save_dir
        save_dir = 'data/cad{}_{}mm_{}x{}_xy_xz_yz/subset{}/{}'.format(CANDIDATES_CSV,OUTPUT_SPACING[1],OUTPUT_DIM,OUTPUT_DIM,subset,'{}')
        if not os.path.exists(save_dir.format('True')):
            os.makedirs(save_dir.format('True'))
        if not os.path.exists(save_dir.format('False')):
            os.makedirs(save_dir.format('False'))

        print '{} - Processing subset'.format(time.strftime("%H:%M:%S")), subset
        image_paths = glob.glob("data/original_lungs/subset{}/*.mhd".format(subset))
        Parallel(n_jobs=4)(delayed(process_image)(image_path,candidates,save_dir) for image_path in image_paths)
        print '{} - Processing subset {} took {} seconds'.format(time.strftime("%H:%M:%S"),subset,np.floor(time.time()-start_time))
        print