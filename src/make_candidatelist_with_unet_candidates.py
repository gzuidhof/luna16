import pandas as pd
import numpy as np
from tqdm import tqdm


# This function checks if a candidate is overlapping with an annotations in the same image
def overlapping(candidate, annotations):
    overlap = False
    canCoords = np.array((candidate.coordX,candidate.coordY,candidate.coordZ))
    # loop over all annotations and check if the xyz-coordinates are in the annotation
    for annIndex in range(0,len(annotations)):
        ann = annotations.iloc[annIndex]
        annCoords = np.array((ann.coordX,ann.coordY,ann.coordZ))
        dAnn = ann.diameter_mm
        # this function calculates the euclidean distance between the candidate and the annotation
        # so we assume that the annotation is always a perfect circle
        if (np.linalg.norm(canCoords-annCoords) < dAnn*0.5):
            overlap = True
    return overlap


# This function checks if a candidate is too close real nodule the parameter "mm"
# determined the range
def DistanceToTPTooClose(candidate, annotations, mm):
    tooClose = False
    canCoords = np.array((candidate.coordX,candidate.coordY,candidate.coordZ))
    # loop over all annotations in the image and check if the xyz-coordinate are too close
	# the the annotaiton
    for annIndex in range(0,len(annotations)):
        ann = annotations.iloc[annIndex]
        annCoords = np.array((ann.coordX,ann.coordY,ann.coordZ))
        if abs(canCoords[0] - annCoords[0]) < mm or abs(canCoords[1] - annCoords[1]) < mm or abs(canCoords[2] - annCoords[2]) < mm:
			return True
	return tooClose


#This function checks if a candidate is lying in an annotation
def overlapping2(candidate, annotation):
    overlap = False
    canCoords = np.array((candidate.coordX,candidate.coordY,candidate.coordZ))
    annCoords = np.array((annotation.coordX,annotation.coordY,annotation.coordZ))
    dAnn = annotation.diameter_mm
    if (np.linalg.norm(canCoords-annCoords) < dAnn*0.5):
        overlap = True
	return overlap

# This function takes a list of candidates and an annotation it checks which candidates
# are lying in the annotation and takes the mean xyz-coordinate of all these candidates
def mergeCandidates(annotation, possibleCandidates):
	x = []
	y = []
	z = []
	found = False
	for indexCan in range(0, len(possibleCandidates)):
		can = possibleCandidates.iloc[indexCan]
		if overlapping2(can, annotation):
			x.append(can.coordX)
			y.append(can.coordY)
			z.append(can.coordZ)
			found = True
	return np.mean(x), np.mean(y), np.mean(z), found

DISTANCE=30

# This fucntion makes the final list, this list exist of TPcandidate and FPcandidates,
# if they are not lying too close to a TPcandidate given in the annotation.csv
def fillTPCandidateList(candidates, annotation, mergedTPCandidatesAndFPCandidates):

	# This part of the function loops over all candidates and checks if they are overlapping
	# with an annotation. If not then the candidate is added to the list, with label 0, if it is
	# lying more than 50 mm from an annotation, given the csv file annotations.csv.
	for indexCan in tqdm(range(0, len(candidates))):
		can = candidates.iloc[indexCan]
		possibleAnnotations = annotation[annotation['seriesuid'] == can.seriesuid]
		if (overlapping(can, possibleAnnotations) == False):
			if (DistanceToTPTooClose(can, possibleAnnotations, DISTANCE) == False):
				mergedTPCandidatesAndFPCandidates = mergedTPCandidatesAndFPCandidates.append({'seriesuid': can.seriesuid, 'coordX': can.coordX, 'coordY': can.coordY, 'coordZ': can.coordZ, 'label': 0}, ignore_index=True)

	# This part of the function adds all the candidates to the list if they are overlapping with
	# an annotation. Before they are added all candidates that are also on lying in the same
	# annotation are merged, by taking the mean.
	for indexAnn in tqdm(range(0,len(annotation))):
		ann = annotation.iloc[indexAnn]
		possibleCandidates = candidates[candidates['seriesuid'] == ann.seriesuid]
		x, y, z, found = mergeCandidates(ann, possibleCandidates)
		if found == True:
			mergedTPCandidatesAndFPCandidates = mergedTPCandidatesAndFPCandidates.append({'seriesuid': ann.seriesuid, 'coordX': x, 'coordY': y, 'coordZ': z, 'label': 1}, ignore_index=True)
	return mergedTPCandidatesAndFPCandidates




if __name__ == "__main__":
    candidates = pd.read_csv('csv/candidates_unet.csv')
    mergedTPCandidatesAndFPCandidates = pd.DataFrame() #is a empty dataFrame
    annotation = pd.read_csv('csv/annotations.csv')
    mergedTPCandidatesAndFPCandidates = fillTPCandidateList(candidates, annotation, mergedTPCandidatesAndFPCandidates)
    mergedTPCandidatesAndFPCandidates.to_csv('LungOnlyMergedUnet89.csv',columns=['seriesuid','coordX','coordY','coordZ','label'])
