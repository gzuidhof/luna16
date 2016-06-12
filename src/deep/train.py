from __future__ import division
import sys
import numpy as np
sys.path.append('./unet')
from unet_trainer import UNetTrainer
from params import params as P
import dataset
from functools import partial
import glob

if __name__ == "__main__":
    np.random.seed(0)
    filenames_train = glob.glob(P.FILENAMES_TRAIN)
    filenames_val = glob.glob(P.FILENAMES_VALIDATION)

    filenames_train = filenames_train[:P.SUBSET]
    filenames_val = filenames_val[:P.SUBSET]

    generator_train = dataset.load_images
    generator_val = partial(dataset.load_images, deterministic=True)

    print "Creating train splits"
    train_splits = dataset.train_splits_by_z(filenames_train, 0.5, P.N_EPOCHS)

    trainer = UNetTrainer()
    trainer.train(train_splits, filenames_val, generator_train, generator_val)
