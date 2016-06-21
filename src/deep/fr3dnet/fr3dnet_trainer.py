from __future__ import division

import numpy as np
import matplotlib
import sys
sys.path.append('../')
sys.path.append('../../')
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
from multiprocessing import Pool
#from multiprocessing.pool import ThreadPool as Pool
import itertools
import util
import functools
import normalize
import augment

def load_data(tup): #filename, coordinates, labels tuple
    size = P.INPUT_SIZE
    data = []
    labels = []

    images = dataset_3D.giveSubImage(tup[0],tup[1],size)
    labels += map(int,tup[2])
    data += images[:]

    data = normalize.normalize(np.array(data, dtype=np.float32))

    if P.ZERO_CENTER:
        data -= P.MEAN_PIXEL

    result = zip([tup[0]]*len(labels), np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32))

    if P.AUGMENT and P.AUGMENTATION_PARAMS['flip']:
        augmentation_extra = []

        for filename, image, label in result:
            if label == 1:
                flipped_images = augment.get_all_flips_3d(image)
                np.random.shuffle(flipped_images)
                flipped_images = flipped_images[:1] #SELECT 1 RANDOM IMAGES OF 7 possible flips
                n_new = len(flipped_images)

                augmentation_extra += zip([filename]*n_new, flipped_images, [label]*n_new)
            else: #For false candidates take one flip combination at random :)
                flip_option = augment.OPTS[np.random.randint(8)]
                augment.flip_given_axes(image, flip_option)

        result += augmentation_extra


    return result

def make_epoch(n, train_true, train_false, val_true, val_false):
    n = n[0]
    train_false = list(train_false)
    val_false = list(val_false)
    np.random.shuffle(train_false)
    np.random.shuffle(val_false)

    n_train_true = len(train_true)
    n_val_true = len(val_true)

    train_epoch = train_true + train_false[:n_train_true*2] #*2 to account for 1 flip directions
    val_epoch = val_true + val_false[:n_val_true*2]

    train_epoch = combine_tups(train_epoch)
    val_epoch = combine_tups(val_epoch)

    print "Epoch {0} n files {1}&{2}".format(n, len(train_epoch), len(val_epoch))
    pool = Pool(processes=12)
    train_epoch_data = list(itertools.chain.from_iterable(pool.imap_unordered(load_data, train_epoch)))
    print "Epoch {0} done loading train".format(n)

    val_epoch_data = list(itertools.chain.from_iterable(pool.imap_unordered(load_data, val_epoch)))
    print "Epoch {0} done loading validation".format(n)
    pool.close()

    np.random.shuffle(train_epoch_data)
    return train_epoch_data, val_epoch_data

def combine_tups(tup):
    names,coords,labels = zip(*tup)
    d = {n:[] for n in names}
    for name,coord,label in tup:
        d[name].append((coord,label))
    data = []
    for name,values in d.iteritems():
        c,l = zip(*values)
        data.append((name,c,l))
    return data

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
        train_fn, val_fn, l_r = fr3dnet.define_updates(net, input_var, target_var)

        self.train_fn = train_fn
        self.val_fn = val_fn
        self.l_r = l_r

    def do_batches(self, fn, batches, metrics):
        batches = list(batches)
        for i, batch in enumerate(tqdm(batches)):
            filenames, inputs, targets = zip(*batch)
            targets = np.array(targets, dtype=np.int32)
            err, l2_loss, acc, predictions = fn(inputs, targets)

            metrics.append([err, l2_loss, acc])
            metrics.append_prediction(targets, predictions)

    def train(self, X_train, X_val):

        train_true = filter(lambda x: x[2]==1, X_train)
        train_false = filter(lambda x: x[2]==0, X_train)

        val_true = filter(lambda x: x[2]==1, X_val)
        val_false = filter(lambda x: x[2]==0, X_val)

        n_train_true = len(train_true)
        n_val_true = len(val_true)

        make_epoch_helper = functools.partial(make_epoch, train_true=train_true, train_false=train_false, val_true=val_true, val_false=val_false)

        logging.info("Starting training...")
        epoch_iterator = ParallelBatchIterator(make_epoch_helper, range(P.N_EPOCHS), ordered=False, batch_size=1, multiprocess=False, n_producers=1)

        for epoch_values in epoch_iterator:
            self.pre_epoch()
            train_epoch_data, val_epoch_data = epoch_values

            train_epoch_data = util.chunks(train_epoch_data, P.BATCH_SIZE_TRAIN)
            val_epoch_data = util.chunks(val_epoch_data, P.BATCH_SIZE_VALIDATION)

            self.do_batches(self.train_fn, train_epoch_data, self.train_metrics)
            self.do_batches(self.val_fn, val_epoch_data, self.val_metrics)

            self.post_epoch()
            logging.info("Setting learning rate to {}".format(P.LEARNING_RATE  * ((0.985)**self.epoch)))
            self.l_r.set_value(P.LEARNING_RATE  * ((0.985)**self.epoch))
