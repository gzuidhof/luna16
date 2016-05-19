import glob
import image_read_write
import blob
import candidates as ca
import pickle
import numpy as np
import os
import evaluate_candidates

def load_data(image_names):
    print "loading images"
    images = []
    origins= []
    spacings = []
    for image in image_names:
        i,o,s = image_read_write.load_itk_image(image)
        images.append(i)
        origins.append(o)
        spacings.append(s)
    print "done"
    return images,origins,spacings

if __name__ == "__main__":
    for subset in xrange(0,1):
        #image_names = glob.glob("../data/subset{}/subset{}/*.mhd".format(subset,subset))
        image_names = glob.glob("../data/subset0/*.mhd")[0:30]
        images,origins,spacings = load_data(image_names)

        blob_images = []
        for index,image in enumerate(images):
            blobs = blob.blob_image(image)

            coords = []
            #coords = [y for y in [x for x in candidates]]

            #slice, blob, xyz

            for slice in blobs:
                for s in slice:
                    coords.append(s)
            #print coords

            #print spacings[index]
            #print coords
            world_coords = np.array([ca.voxel_2_world(y,origins[index],spacings[index]) for y in coords])
            #print world_coords
            #print world_coords
            name = os.path.split(image_names[index])[1].replace('.mhd','')
            #print name
            #print np.array(world_coords).shape
            candidates = ca.coords_to_candidates(world_coords, name)
            #print candidates

            #candidates = ca.merge_candidates(candidates)
            #print len(candidates)

            #image_read_write.save_candidates('../data/hoi_candidates.csv', candidates)
            evaluate_candidates.run(candidates)
