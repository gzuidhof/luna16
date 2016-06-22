import candidates as ca
import numpy as np


annotations = ca.load_candidates("../data/annotations.csv")
#found_candidates = np.zeros((len(annotations),1))
found_candidates = []
for i in xrange (0,len(annotations)):
    found_candidates.append([])

nr_candidates = 0
nr_annotations = 0

def check_coordinates(image_coord,candidate):
    diameter = candidate[4]
    coord_x = candidate[1]
    coord_y = candidate[2]
    coord_z = candidate[3]
    coords = np.array((coord_x,coord_y,coord_z))
    image_coord = np.array((image_coord[0],image_coord[1],image_coord[2]))

    #print np.linalg.norm(image_coord-coords)

    if np.linalg.norm(image_coord-coords) < diameter*0.5:
        return True
    return False

def check_coordinates2(image_coord,candidate):
    diameter = candidate[3]
    coord_x = candidate[0]
    coord_y = candidate[1]
    coord_z = candidate[2]
    coords = np.array((coord_x,coord_y,coord_z))
    image_coord = np.array((image_coord[0],image_coord[1],image_coord[2]))

    #print np.linalg.norm(image_coord-coords)

    if np.linalg.norm(image_coord-coords) < diameter*0.5:
        return True
    return False


def is_candidate(image_coord,image_annotations):
    #print annotations['seriesuid']
    #print image_name

    #print "Amount of actual nodules:",len(image_annotations.values)
    #if len(image_annotations.values) > 0:
    #    exit()

    for candidate in image_annotations.values:
        if check_coordinates(image_coord,candidate):
            return candidate
    return False

def is_candidate2(image_coord,image_annotations):
    #print annotations['seriesuid']
    #print image_name

    # print "Amount of actual nodules:",len(image_annotations.values)
    #if len(image_annotations.values) > 0:
    #    exit()

    for candidate in image_annotations:
        if check_coordinates2(image_coord,candidate):
            return candidate
    return False


def evaluate(train_candidates):
    global annotations
    #found_candidates = 0
    #print found_candidates.shape
    global nr_annotations
    global nr_candidates
    image_annotations = annotations[annotations['seriesuid'] == train_candidates[0]["image_name"]]
    nr_annotations += len(image_annotations)
    nr_candidates += len(train_candidates)
    for candidate in train_candidates:
        #print candidate
        can =  is_candidate(candidate["image_coord"],image_annotations)
        if can is not False:
            index = np.where(annotations.values==can.all())[0]
            found_candidates[index].append([can[1],can[2],can[3]])
            nr_candidates -= 1

    # for ann in image_annotations.values:
    #     index = np.where(annotations.values==ann.all())[0]
    #     if found_candidates[index] == 0:
    #         print ann

    found = 0
    for entry in found_candidates:
        if entry != []:
            found+=1
    if nr_annotations != 0:
         print "recall",float(found)/nr_annotations
         print "precision",float(found)/nr_candidates


#In found_candidates there can be multiple entries that belong to the same annotation, these have to be averaged to find the center of the blob.
def save_mean_candidates():
    for entry in found_candidates:
        if entry != []:
            coords = np.mean(entry,axis=0)
            print coords


def run(candidates):

    #annotations = candidates.load_candidates("../data/annotations/annotations.csv")
    #print is_candidate([-130,-177,-299],"1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860")
    #candidates = candidates.load_candidates("../data/annotations/candidates.csv")
    #candidates = candidates.load_candidates("../data/hoi_candidates.csv")
    train_candidates = []
    name = candidates.values[0][0]
    all_cands = []
    # print candidates
    for object in candidates.values:
        if object[0] == name:
            train_candidates.append({"image_name":object[0],"image_coord":[object[1],object[2],object[3]]})
        else:
            name = object[0]
            all_cands.append(train_candidates)
            train_candidates = []
            train_candidates.append({"image_name":object[0],"image_coord":[object[1],object[2],object[3]]})
    for image in all_cands:
        evaluate(image)



if __name__ == "__main__":
    global annotations
    #annotations = candidates.load_candidates("../data/annotations/annotations.csv")
    #print is_candidate([-130,-177,-299],"1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860")
    #candidates = candidates.load_candidates("../data/annotations/candidates.csv")

    annotations = ca.load_candidates("../data/annotations.csv")
    candidates = ca.load_candidates("../data/hoi_candidates.csv")
    train_candidates = []
    for object in candidates.values:
        train_candidates.append({"image_name":object[0],"image_coord":[object[1],object[2],object[3]]})
    evaluate(train_candidates)
