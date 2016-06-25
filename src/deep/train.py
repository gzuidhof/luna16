from __future__ import division
import sys
import numpy as np
from params import params as P
import pandas as pd
sys.path.append('../')
import subset



if P.ARCHITECTURE == 'unet':
    sys.path.append('./unet')
    from unet_trainer import UNetTrainer
    import dataset
elif P.ARCHITECTURE == 'resnet':
    sys.path.append('./resnet')
    from resnet_trainer import ResNetTrainer
    import dataset_2D

elif P.ARCHITECTURE == 'fr3dnet':
    sys.path.append('./fr3dnet')
    from fr3dnet_trainer import Fr3dNetTrainer
    import dataset_3D
from functools import partial
import glob



def fr3dnet_dataset(subset_nr,df,name_per_subset):
    train_x_names = name_per_subset[subset_nr]
    cands = df[df['seriesuid'].isin(train_x_names)]
    coords = zip(cands.values[:,1],cands.values[:,2],cands.values[:,3])
    names = cands.values[:,0]
    labels = cands.values[:,4]
    path_names = [P.DATA_FOLDER + 'subset{0}/{1}.mhd'.format(subset_nr,y) for y in names]
    return zip(path_names,coords,labels)
if __name__ == "__main__":

    np.random.seed(0)

    if P.ARCHITECTURE != "fr3dnet":
        filenames_train = glob.glob(P.FILENAMES_TRAIN)
        filenames_val = glob.glob(P.FILENAMES_VALIDATION)

        filenames_train = filenames_train[:P.SUBSET]
        filenames_val = filenames_val[:P.SUBSET]

    if P.ARCHITECTURE == 'unet':

        generator_train = dataset.load_images
        generator_val = partial(dataset.load_images, deterministic=True)

        print "Creating train splits"
        train_splits = dataset.train_splits_by_z(filenames_train, 0.3, P.N_EPOCHS)

        trainer = UNetTrainer()
        trainer.train(train_splits, filenames_val, generator_train, generator_val)

    elif P.ARCHITECTURE == 'resnet':
        from functools import partial

        X_train = glob.glob(P.FILENAMES_TRAIN)
        X_val = glob.glob(P.FILENAMES_VALIDATION)

        print "N Train", len(X_train)
        print "N Val", len(X_val)

        assert len(X_train)+len(X_val) != 0

        train_generator = dataset_2D.load_images
        validation_generator = partial(dataset_2D.load_images, deterministic=True)

        trainer = ResNetTrainer()
        trainer.train(train_generator, X_train, validation_generator, X_val)

    elif P.ARCHITECTURE == "fr3dnet":
        df = pd.read_csv("../../data/candidates_v2.csv")
        name_per_subset = subset.get_subset_to_filename_dict()
        train_x = []
        subsets_train = map(int,P.FILENAMES_TRAIN.split(','))
        for s in subsets_train:
            train_x += fr3dnet_dataset(s,df,name_per_subset)

        val_x = []
        subsets_val = map(int,P.FILENAMES_VALIDATION.split(','))
        for s in subsets_val:
            val_x += fr3dnet_dataset(s,df,name_per_subset)
        trainer = Fr3dNetTrainer()
        trainer.train(train_x,val_x)
