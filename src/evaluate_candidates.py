import candidates as ca
import numpy as np


annotations = ca.load_candidates("../../CSVFILES/CSVFILES/annotations.csv")
found_candidates = np.zeros((len(annotations),1))
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
    # if image_coord[0] > coord_x - 0.5*diameter and image_coord[0] < coord_x + 0.5*diameter and\
    #    image_coord[1] > coord_y - 0.5*diameter and image_coord[1] < coord_y + 0.5*diameter and\
    #    image_coord[2] > coord_z - 0.5*diameter and image_coord[2] < coord_z + 0.5*diameter:
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
            found_candidates[index] = 1

    if nr_annotations != 0:
        print "recall",float(np.sum(found_candidates))/nr_annotations
        print "precision",float(np.sum(found_candidates))/nr_candidates



def run(candidates):

    #annotations = candidates.load_candidates("../data/annotations/annotations.csv")
    #print is_candidate([-130,-177,-299],"1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860")
    #candidates = candidates.load_candidates("../data/annotations/candidates.csv")
    #candidates = candidates.load_candidates("../data/hoi_candidates.csv")
    train_candidates = []

    #print candidates
    for object in candidates.values:
        train_candidates.append({"image_name":object[0],"image_coord":[object[3],object[2],object[1]]})
    evaluate(train_candidates)


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
