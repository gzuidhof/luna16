from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
import pandas as pd
from tqdm import tqdm
import blob
import pickle
import glob
import os
import scipy.misc
from skimage.io import imread
from skimage import morphology
from scipy import ndimage
import cv2
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass



from image_read_write import load_itk_image

CANDIDATES_COLUMNS = ['seriesuid','coordX','coordY','coordZ','label']
THRESHOLD = 225



def unet_candidates():
    cands = glob.glob("../data/predictions_epoch9_23_all/*.png")
    #df = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','class'])
    data = []
    imname = ""
    origin = []
    spacing = []
    nrimages = 0
    for name in tqdm(cands):

        #image = imread(name)
        image_t = imread(name)
        image_t = image_t.transpose()
        #Thresholding
        image_t[image_t<THRESHOLD] = 0
        image_t[image_t>0] = 1
        #erosion
        selem = morphology.disk(1)
        image_eroded = image_t
        image_eroded = morphology.binary_erosion(image_t,selem=selem)

        label_im, nb_labels = ndimage.label(image_eroded)
        imname3 = os.path.split(name)[1].replace('.png','')

        splitted = imname3.split("slice")
        slice = splitted[1]
        imname2 = splitted[0][:-1]
        centers = []
        for i in xrange(1,nb_labels+1):
            blob_i = np.where(label_im==i,1,0)
            mass = center_of_mass(blob_i)
            centers.append([mass[1],mass[0]])


        if imname2 != imname:
            if os.path.isfile("../data/1_1_1mm_512_x_512_annotation_masks/spacings/{0}.pickle".format(imname2)):
                with open("../data/1_1_1mm_512_x_512_annotation_masks/spacings/{0}.pickle".format(imname2), 'rb') as handle:
                    dic = pickle.load(handle)
                    origin = dic["origin"]
                    spacing = dic["spacing"]

            imname = imname2
            nrimages +=1

        for center in centers:
            coords = voxel_2_world([int(slice),center[1]+(512-324)*0.5,center[0]+(512-324)*0.5],origin,spacing)
            data.append([imname2,coords[2],coords[1],coords[0],'?'])

        #if nrimages == 5:
        #    break

    df = pd.DataFrame(data,columns=CANDIDATES_COLUMNS)
    save_candidates("../data/candidates_unet_final_23.csv",df)


def candidates_to_image(cands,radius):
    image_names = []
    for subset in xrange(0,10):
        subset_names = glob.glob("../data/subset{0}/*.mhd".format(subset))
        names = [os.path.split(x)[1].replace('.mhd','') for x in subset_names]
        image_names.append(names)
    previous_candidate = ""
    images = []
    image = []
    origin = []
    spacing = []
    number = 0
    for candidate in tqdm(cands.values):
        if candidate[0] != previous_candidate:
            number = 0
            previous_candidate = candidate[0]
            for image_subset in xrange(0,10):
                if candidate[0] in image_names[image_subset]:
                    image,origin,spacing = load_itk_image("../data/subset{0}/{1}.mhd".format(image_subset,candidate[0]))
                    break
        coords = world_2_voxel([candidate[3],candidate[2],candidate[1]],origin,spacing)
        im = image_part_from_candidate(image,coords,radius)
        #images.append(im)
        if candidate[4]:
            label = "true"
        else:
            label = "false"
        scipy.misc.imsave('../data/samples/{0}_{1}_{2}.jpg'.format(candidate[0],number,label), im)
        number += 1
    return images

def image_part_from_candidate(image,coords,radius):
    im = np.zeros((radius*2,radius*2))
    for x in xrange(-radius,radius):
        for y in xrange(-radius,radius):
                try:
                    im[x+radius,y+radius]=image[coords[0],coords[1]+x,coords[2]+y]
                except:
                    im[x+radius,y+radius]=-1000
    return im


#Merge candidates of a single scan
def merge_candidates_scan(candidates, seriesuid, distance=5.):
    distances = pdist(candidates, metric='euclidean')
    adjacency_matrix = squareform(distances)

    # Determine nodes within distance, replace by 1 (=adjacency matrix)
    adjacency_matrix = np.where(adjacency_matrix<=distance,1,0)

    # Determine all connected components in the graph
    n, labels = connected_components(adjacency_matrix)
    new_candidates = np.zeros((n,3))

    # Take the mean for these connected components
    for cluster_i in range(n):
        points = candidates[np.where(labels==cluster_i)]
        center = np.mean(points,axis=0)
        new_candidates[cluster_i,:] = center

    x = new_candidates[:,0]
    y = new_candidates[:,1]
    z = new_candidates[:,2]
    labels = [seriesuid]*len(x)
    class_name = [0]*len(x)

    data= zip(labels,x,y,z,class_name)

    new_candidates = pd.DataFrame(data,columns=CANDIDATES_COLUMNS)

    return new_candidates

def merge_candidates(df_candidates, distance=5.):
    new_candidates = pd.DataFrame()
    for scan_name in tqdm(df_candidates['seriesuid'].unique()):
        #print "Merging scan", scan_name
        df = df_candidates[df_candidates['seriesuid']==scan_name]
        x = df['coordX']
        y = df['coordY']
        z = df['coordZ']
        shape = (len(x),3)
        candidates = np.zeros(shape)
        candidates[:,0]=x
        candidates[:,1]=y
        candidates[:,2]=z

        new = merge_candidates_scan(candidates,seriesuid=scan_name,distance=distance)
        new_candidates = new_candidates.append(new)


    #print new_candidates
    return new_candidates


def world_2_voxel(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def voxel_2_world(voxelCoord, origin, spacing):
    stretchedVoxelCoord = voxelCoord * spacing
    worldCoord = stretchedVoxelCoord + origin
    return worldCoord

def load_candidates(filename, as_coords=False):
    candidates = pd.read_csv(filename)
    return candidates

# Save candidates given filename and pandas dataframe
# Dataframe with columns:
# seriesuid, coordX, coordY, coordZ, class
# class seems to be 0 always
def save_candidates(filename, df_candidates):
    df_candidates.to_csv(filename, index=False)

def coords_to_candidates(coords, seriesuid):
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    names = [seriesuid]*len(x)
    class_name = [0]*len(x)

    data = zip(names,x,y,z,class_name)
    candidates = pd.DataFrame(data,columns=CANDIDATES_COLUMNS)
    return candidates

if __name__ == "__main__":

    unet_candidates()
    quit()
    df = load_candidates('../data/candidates.csv')
    images = candidates_to_image(df,15)
    #new_candidates = merge_candidates(df)
    #save_candidates('test.csv', new_candidates)


    #coords = blob.blob_image('../data/hoi.mhd')
    #with open('../data/hoi_coords.pkl','w') as f:
    #    pickle.dump(coords, f)
    with open('../data/hoi_coords.pkl','r') as f:
        candidates = pickle.load(f)

    coords = []
    #coords = [y for y in [x for x in candidates]]

    #slice, blob, xyz

    for slice in candidates:
        #print slice
        for blob in slice:
            coords.append(blob)
    #print coords

    image, origin, spacing = load_itk_image('../data/hoi.mhd')

    world_coords = np.array([voxel_2_world(y,origin,spacing) for y in coords])

    #print world_coords


    candidates = coords_to_candidates(world_coords, '1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260')
    print len(candidates)
    candidates = merge_candidates(candidates)
    print len(candidates)

    save_candidates('../data/hoi_candidates.csv', candidates)
