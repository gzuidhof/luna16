import csv
import glob
import os
import numpy as np
from collections import defaultdict
import candidates as ca
import image_read_write
from pandas import DataFrame as df
import pandas as pd
import candidates
import make_candidatelist_with_unet_candidates as mcwuc

def merge_pipe_and_unet(unetoutput, annotations, size):
    candidates = pd.read_csv(unetoutput)
    mergedTPCandidatesAndFPCandidates = pd.DataFrame() #is a empty dataFrame
    annotation = pd.read_csv(annotations)
    mergedTPCandidatesAndFPCandidates = mcwuc.fillTPCandidateList(candidates, annotation, mergedTPCandidatesAndFPCandidates)
    mergedTPCandidatesAndFPCandidates.to_csv('candidates_unet_TF_merged.csv', index=False)

def save_to_csv(data_per_slice, imagename):
    dataframe = df(data_per_slice)
    dataframe.to_csv('E:\uni\Medical\Project\CSVFILES\CSVFILES\\removeLungs\\' + imagename + 'removedLungs.csv', index=False)

def merge_candidates(csvfile, size):
    # initialize een map
    information_per_slice = defaultdict(list)
    with open(csvfile, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)
        for rows in reader:
            information_per_slice[rows[0]].append(map(float, rows[1:4]))
    for imagename, data_per_slice in information_per_slice.items():
        data_per_slice = remove_pipeline(data_per_slice, imagename)
        print(data_per_slice)
        data_per_slice = np.array(data_per_slice)
        if(len(data_per_slice) > 0):
            data_per_slice = candidates.merge_candidates_scan(data_per_slice, seriesuid=imagename, distance=size)
            save_to_csv(data_per_slice, imagename)


def remove_pipeline(data_per_slice, imagename):
    # print 'reading for ' + imagename
    lungmasklocation = 'E:\uni\Medical\Project\seg-lungs-LUNA16\seg-lungs-LUNA16\\' + imagename + '.mhd'
    numpyImage, numpyOrigin, numpySpacing = image_read_write.load_itk_image(lungmasklocation)
    to_remove = []
    for coords in data_per_slice:
        voxel = ca.world_2_voxel(coords[::-1], numpyOrigin, numpySpacing)
        if((numpyImage[voxel[0]][voxel[1]][voxel[2]] != 4) and (numpyImage[voxel[0]][voxel[1]][voxel[2]] != 3)):
            to_remove.append(coords)
    for coords in to_remove:
        data_per_slice.remove(coords)
    return data_per_slice

# candidates = 'E:\uni\Medical\Project\CSVFILES\CSVFILES\candidates_unet_final_45.csv'
# annotations = 'E:/uni/Medical/Project/CSVFILES/CSVFILES/annotations.csv'
# merge_pipe_and_unet(candidates,annotations, 2)
merge_candidates('E:\uni\Medical\Project\luna16\src\candidates_unet_TF_merged.csv', 2)