from __future__ import division
import sys
import numpy as np
from params import params as P


if P.ARCHITECTURE == 'unet':
    sys.path.append('./unet')
    from unet_trainer import UNetTrainer
    import dataset
elif P.ARCHITECTURE == 'resnet':
    sys.path.append('./resnet')
    from resnet_trainer import ResNetTrainer
    import dataset_2D
from functools import partial
import glob

if __name__ == "__main__":
    np.random.seed(0)
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


        X_train = glob.glob(P.FILENAMES_TRAIN)
        X_val = glob.glob(P.FILENAMES_VALIDATION)

        train_generator = dataset_2D.load_images
        validation_generator = dataset_2D.load_images

        trainer = ResNetTrainer()
        trainer.train(train_generator, X_train, validation_generator, X_val)
