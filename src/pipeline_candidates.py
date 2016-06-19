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
        candidates = ca.load_candidates("E:\uni\Medical\Project\CSVFILES\CSVFILES\\removeLungs\\1.3.6.1.4.1.14519.5.2.1.6279.6001.100398138793540579077826395208removedLungs.csv",False)
        # candidates = ca.load_candidates("../data/candidates_unet_final.csv",False)
        #candidates = ca.load_candidates('../data/candidates_unet_TF_merged_FP_removed_if_close_45.csv', False)
        # print candidates
        #candidates = ca.merge_candidates(candidates,distance=1.337)
        evaluate_candidates.run(candidates)
        #evaluate_candidates.save_mean_candidates()
        quit()
