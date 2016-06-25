import glob
import image_read_write
import blob
import candidates as ca
import pickle
import numpy as np
import os
import evaluate_candidates
import pandas

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

        candidates = ca.load_candidates("../data/unet_candidates25/unetNew3.csv",False)

        #candidates = ca.load_candidates("../data/unet_candidates_noclose3/unetNew.csv",False)
        print len(candidates)
        #candidates = ca.merge_candidates(candidates,distance=0.)
        print len(candidates)
        evaluate_candidates.run(candidates)
        #evaluate_candidates.save_mean_candidates()
        quit()
        #image_names = glob.glob("../data/subset{}/*.mhd".format(subset,subset))
        image_names = glob.glob("data/subset0/*.mhd")
        images,origins,spacings = load_data(image_names)

        blob_images = []
        for index,image in enumerate(images):
            blobs = blob.blob_image_multiscale2(image,type=2)

            coords = []
            #coords = [y for y in [x for x in candidates]]

            #slice, blob, xyz

            for slice in blobs:
                for s in slice:
                    coords.append(s)
            world_coords = np.array([ca.voxel_2_world(y[0:3],origins[index],spacings[index]) for y in coords])

            name = os.path.split(image_names[index])[1].replace('.mhd','')

            candidates = ca.coords_to_candidates(world_coords, name)

            #candidates = ca.merge_candidates(candidates)
            #print len(candidates)
            ca.save_candidates("data/blob_candidates/{0}.csv".format(name), candidates)
            #image_read_write.save_candidates('../data/blob_candidates/', candidates)

            evaluate_candidates.run(candidates)

    # os.chdir('E:\uni\Medical\Project\CSVFILES\CSVFILES\\removeLungs')
    # for file in glob.glob('*.csv'):
    # file = 'E:\uni\Medical\Project\CSVFILES\CSVFILES\\removeLungs\unet89\LungOnlyMergedUnet89.csv'
    # candidates = ca.load_candidates(file, False)
        # print candidates
    # evaluate_candidates.run(candidates)
    # for subset in xrange(0,1):
    file = 'E:\uni\Medical\Project\luna16\src\candidates_unet_TF_merged.csv'
    candidates = ca.load_candidates(file,False)
        # candidates = ca.load_candidates("../data/candidates_unet_final.csv",False)
        # candidates = ca.load_candidates('../data/candidates_unet_TF_merged_FP_removed_if_close_45.csv', False)
        # print candidates
        # candidates = ca.merge_candidates(candidates,distance=1.337)
    evaluate_candidates.run(candidates)
        # evaluate_candidates.save_mean_candidates()
        # quit()
