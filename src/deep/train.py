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

    trainer = UNetTrainer()
    trainer.train(filenames_train, filenames_val, generator_train, generator_val)
