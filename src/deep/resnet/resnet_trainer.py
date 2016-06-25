from __future__ import division
import time
import numpy as np
import trainer
from params import params as P
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
import logging
import scipy.misc
from parallel import ParallelBatchIterator
from tqdm import tqdm
import os.path

import dataset_2D

import theano
import theano.tensor as T
import lasagne
import resnet
from resnet import LR_SCHEDULE
import augment


class ResNetTrainer(trainer.Trainer):
    def __init__(self):
        metric_names = ['Loss','L2','Accuracy']
        super(ResNetTrainer, self).__init__(metric_names)

        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        logging.info("Defining network")
        net = resnet.ResNet_FullPre_Wide(input_var, P.DEPTH, P.BRANCHING_FACTOR)
        self.network = net
        train_fn, val_fn, l_r = resnet.define_updates(self.network, input_var, target_var)

        self.train_fn = train_fn
        self.val_fn = val_fn
        self.l_r = l_r

    def do_batches(self, fn, batch_generator, metrics):
        for i, batch in enumerate(tqdm(batch_generator)):
            inputs, targets = batch
            targets = np.array(np.argmax(targets, axis=1), dtype=np.int32)
            err, l2_loss, acc, prediction, _ = fn(inputs, targets)

            metrics.append([err, l2_loss, acc])
            metrics.append_prediction(targets, prediction)

    def train(self, generator_train, X_train, generator_val, X_val):
        #filenames_train, filenames_val = patch_sampling.get_filenames()
        #generator = partial(patch_sampling.extract_random_patches, patch_size=P.INPUT_SIZE, crop_size=OUTPUT_SIZE)


        train_true = filter(lambda x: "True" in x, X_train)
        train_false = filter(lambda x: "False" in x, X_train)

        print "N train true/false", len(train_true), len(train_false)
        print X_train[:2]

        val_true = filter(lambda x: "True" in x, X_val)
        val_false = filter(lambda x: "False" in x, X_val)

        n_train_true = len(train_true)
        n_val_true = len(val_true)

        logging.info("Starting training...")
        for epoch in range(P.N_EPOCHS):
            self.pre_epoch()

            if epoch in LR_SCHEDULE:
                logging.info("Setting learning rate to {}".format(LR_SCHEDULE[epoch]))
                self.l_r.set_value(LR_SCHEDULE[epoch])


            np.random.shuffle(train_false)
            np.random.shuffle(val_false)

            train_epoch_data = train_true + train_false[:n_train_true]
            val_epoch_data = val_true + val_false[:n_val_true]

            np.random.shuffle(train_epoch_data)
            #np.random.shuffle(val_epoch_data)

            #Full pass over the training data
            train_gen = ParallelBatchIterator(generator_train, train_epoch_data, ordered=False,
                                                batch_size=P.BATCH_SIZE_TRAIN//3,
                                                multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                                n_producers=P.N_WORKERS_LOAD_AUGMENTATION)

            self.do_batches(self.train_fn, train_gen, self.train_metrics)

            # And a full pass over the validation data:
            val_gen = ParallelBatchIterator(generator_val, val_epoch_data, ordered=False,
                                                batch_size=P.BATCH_SIZE_VALIDATION//3,
                                                multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                                n_producers=P.N_WORKERS_LOAD_AUGMENTATION)

            self.do_batches(self.val_fn, val_gen, self.val_metrics)
            self.post_epoch()

if __name__ == "__main__":
    X_train = glob.glob(P.FILENAMES_TRAIN)
    X_val = glob.glob(P.FILENAMES_VALIDATION)

    train_generator = dataset_2D.load_images
    validation_generator = dataset_2D.load_images

    trainer = ResNetTrainer()
    trainer.train(train_generator, X_train, validation_generator, X_val)
