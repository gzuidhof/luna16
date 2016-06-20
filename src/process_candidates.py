from __future__ import division
import numpy as np
import image_read_write
import glob
import candidates
import os
import create_lung_segmented_same_spacing_data
import matplotlib.pyplot as plt

def draw_circles(image,cands,origin,spacing):

    image_mask = np.zeros(image.shape)
    for ca in cands.values:
        diameter = int(np.ceil(ca[4]/2))
        #diameter = 5
        coord_x = ca[3]
        coord_y = ca[2]
        coord_z = ca[1]
        #print diameter

        image_coord = np.array((coord_x,coord_y,coord_z))
        #print ca
        image_coord = candidates.world_2_voxel(image_coord,origin,spacing)
        #print image_coord
        #print image_coord

        #print np.linalg.norm(image_coord-coords)

        for x in xrange(-diameter,diameter):
            for y in xrange(-diameter,diameter):
                for z in xrange(-diameter,diameter):
                    coords = candidates.world_2_voxel(np.array((coord_x+x,coord_y+y,coord_z+z)),origin,spacing)
                    if np.linalg.norm(image_coord-coords) < diameter:
                        image_mask[coords[0],coords[1],coords[2]] = 1

    print np.sum(image_mask)
    # for i in xrange(0,300):
    #     if np.sum(image_mask[i,:,:])>5:
    #         plt.imshow(image_mask[150,:,:])
    #         plt.show()
    #         break
    return image_mask
    #image_read_write.save_itk(image_mask,"../data/circle_sample.mhd")

if __name__ == "__main__":
    for i in xrange(0,2):
        cads = candidates.load_candidates("/data/annotations.csv")
        #index = 2
        image_names = glob.glob("../data/subset{}/*.mhd".format(i))
        for image_name in image_names:
            image,origin,spacing = image_read_write.load_itk_image(image_name)
            name = os.path.split(image_name)[1].replace('.mhd','')
            image_cads = cads[cads['seriesuid'] == name]
            print name
            image_mask=draw_circles(image,image_cads,origin,spacing)
            create_lung_segmented_same_spacing_data.save_itk(image_mask,origin,spacing,"../data/annotation_masks/subset{}/{}.mhd".format(i,name))
