from __future__ import division

import numpy as np
import matplotlib
import sys
sys.path.append('../')
import trainer
from params import params as P
import fr3dnet
matplotlib.use('Agg')
import logging
from parallel import ParallelBatchIterator
from tqdm import tqdm
import theano
import dataset_3D
import theano.tensor as T


class Fr3dNetTrainer(trainer.Trainer):
    def __init__(self):
        metric_names = ['Loss','L2','Accuracy']
        super(Fr3dNetTrainer, self).__init__(metric_names)

        tensor5 = T.TensorType(theano.config.floatX, (False,) * 5)
        input_var = tensor5('inputs')
        target_var = T.ivector('targets')

        logging.info("Defining network")
        net = fr3dnet.define_network(input_var)
        self.network = net
        train_fn, val_fn = fr3dnet.define_updates(net, input_var, target_var)

        self.train_fn = train_fn
        self.val_fn = val_fn

    def do_batches(self, fn, batch_generator, metrics):
        for i, batch in enumerate(tqdm(batch_generator)):
            inputs, targets = batch
            targets = np.array(targets, dtype=np.int32)
            err, l2_loss, acc = fn(inputs, targets)

            metrics.append([err, l2_loss, acc])
            #metrics.append_prediction(true, prob_b)

    def train(self, X_train, X_val):
        #filenames_train, filenames_val = patch_sampling.get_filenames()
        #generator = partial(patch_sampling.extract_random_patches, patch_size=P.INPUT_SIZE, crop_size=OUTPUT_SIZE)


        def load_data(tup):
            size = P.INPUT_SIZE
            data = []
            labels = []
            for t in tup:
                image = dataset_3D.giveSubImage(t[0],[t[1]],size)
                #image = np.expand_dims(image, axis=0)
                labels.append(int(t[2]))
                data.append(image)
            return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32)

        train_true = filter(lambda x: x[2]==1, X_train)
        train_false = filter(lambda x: x[2]==0, X_train)

        val_true = filter(lambda x: x[2]==1, X_val)
        val_false = filter(lambda x: x[2]==0, X_val)

        n_train_true = len(train_true)
        n_val_true = len(val_true)


        logging.info("Starting training...")
        for epoch in range(P.N_EPOCHS):
            self.pre_epoch()

            np.random.shuffle(train_false)
            np.random.shuffle(val_false)

            train_epoch_data = train_true + train_false[:n_train_true]
            val_epoch_data = val_true + val_false

            np.random.shuffle(train_epoch_data)
            #np.random.shuffle(val_epoch_data)

            #Full pass over the training data
            train_gen = ParallelBatchIterator(load_data, train_epoch_data, ordered=False,
                                                batch_size=P.BATCH_SIZE_TRAIN,
                                                multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                                n_producers=P.N_WORKERS_LOAD_AUGMENTATION)

            self.do_batches(self.train_fn, train_gen, self.train_metrics)

            # And a full pass over the validation data:
            val_gen = ParallelBatchIterator(load_data, val_epoch_data, ordered=False,
                                                batch_size=P.BATCH_SIZE_VALIDATION,
                                                multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                                n_producers=P.N_WORKERS_LOAD_AUGMENTATION)

            self.do_batches(self.val_fn, val_gen, self.val_metrics)
            self.post_epoch()
