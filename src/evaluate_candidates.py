import image_read_write


global candidates

def check_coordinates(image_coord,candidate):
    diameter = candidate[4]
    coord_x = candidate[1]
    coord_y = candidate[2]
    coord_z = candidate[3]
    if image_coord[0] > coord_x - 0.5*diameter and image_coord[0] < coord_x + 0.5*diameter and\
       image_coord[1] > coord_y - 0.5*diameter and image_coord[1] < coord_y + 0.5*diameter and\
       image_coord[2] > coord_z - 0.5*diameter and image_coord[2] < coord_y + 0.5*diameter:
        return True
    return False


def is_candidate(image_coord,image_name):
    image_candidates = candidates[candidates['seriesuid'] == image_name]
    #print image_candidates
    for candidate in image_candidates.values:
        print candidate
        if check_coordinates(image_coord,candidate):
            return True
    return False

if __name__ == "__main__":
    global candidates
    candidates = image_read_write.load_candidates("../data/annotations/annotations.csv")
    print is_candidate([-130,-177,-299],"1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860")



