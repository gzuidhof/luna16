import glob
import image_read_write
import blob
import candidates
import pickle
import numpy as np
import os
import evaluate_candidates

def load_data(image_names):
    print "loading images"
    images = origins= spacings = []
    for image in image_names:
        i,o,s = image_read_write.load_itk_image(image)
        images.append(i)
        origins.append(o)
        spacings.append(s)
    "done"
    return images,origins,spacings

if __name__ == "__main__":
    for subset in xrange(0,10):
        image_names = glob.glob("../data/subset{}/subset{}/*.mhd".format(subset,subset))
        images,origins,spacings = load_data(image_names[0])

        blob_images = []
        for index,image in enumerate(images):
            candidates = blob.blob_image(image_names[index])

        coords = []
        #coords = [y for y in [x for x in candidates]]

        #slice, blob, xyz

        for slice in candidates:
            #print slice
            for blob in slice:
                coords.append(blob)
        #print coords


        world_coords = np.array([candidates.voxel_2_world(y,origins[index],spacings[index]) for y in candidates])

        #print world_coords
        name = os.path.split(image_names[index])
        print name[1].replace(".mhd","")
        candidates = candidates.coords_to_candidates(world_coords, name)
        print len(candidates)
        candidates = candidates.merge_candidates(candidates)
        print len(candidates)

        #image_read_write.save_candidates('../data/hoi_candidates.csv', candidates)
        evaluate_candidates.run(candidates)


