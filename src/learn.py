from __future__ import division
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.misc

if __name__ == "__main__":
    from dataset import load_images
    import theano
    import theano.tensor as T
    import lasagne
    import unet
    from unet import INPUT_SIZE, OUTPUT_SIZE

from tqdm import tqdm
from glob import glob

import cPickle as pickle
from parallel import ParallelBatchIterator

def calc_dice(train,truth):
    score = np.sum(train[truth>0])*2.0 / (np.sum(train) + np.sum(truth))
    return score

def make_dir_if_not_present(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":


    # create Theano variables for input and target minibatch
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets', dtype='int64')

    print "Defining network"
    net_dict = unet.define_network(input_var)
    network = net_dict['out']

    train_fn, val_fn = unet.define_updates(network, input_var, target_var)

    model_name = 'unet'+str(int(time.time()))
    model_folder = os.path.join('../data/models',model_name)
    plot_folder = os.path.join('../images/plot',model_name)

    folders = ['../images','../images/plot','../data','../data/models', model_folder, plot_folder]
    map(make_dir_if_not_present, folders)


    np.random.seed(1)
    folder_train = './../data/1_1_1mm_512_x_512_lung_slices/subset[0-2]/'
    filenames_train = glob(folder_train+ '*.pkl.gz')
    #print filenames
    folder_val = './../data/1_1_1mm_512_x_512_lung_slices/subset[3]/'
    filenames_val = glob(folder_val+ '*.pkl.gz')

    np.random.shuffle(filenames_train)
    np.random.shuffle(filenames_val)

    filenames_train = filenames_train[:1]
    filenames_val = filenames_train[:10]

    train_batch_size = 1
    val_batch_size = 2

    num_epochs = 400

    metric_names = ['loss  ','accuracy','l2    ','dice  ','precision','recall']
    train_metrics_all = []
    val_metrics_all = []

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_batches = 0
        start_time = time.time()

        np.random.shuffle(filenames_train)
        train_gen = ParallelBatchIterator(load_images, filenames_train, ordered=True, batch_size=train_batch_size, multiprocess=False)

        train_metrics = []
        val_metrics = []

        #for batch in iterate_minibatches(X_train, y_train, 1, shuffle=True):
        for i, batch in enumerate(tqdm(train_gen)):
            inputs, targets = batch
            err, acc, l2_loss, true, prob, dice, tp,tn,fp,fn= train_fn(inputs, targets)

            train_metrics.append([err, acc, l2_loss, dice, tp, tn, fp, fn])
            train_batches += 1

            if np.ceil(i//train_batch_size) == 0:
                im = np.hstack((
                    true[:OUTPUT_SIZE**2].reshape(OUTPUT_SIZE,OUTPUT_SIZE),
                    prob[:OUTPUT_SIZE**2].reshape(OUTPUT_SIZE,OUTPUT_SIZE)))

                plt.imsave(os.path.join(plot_folder,'train_{}_epoch{}.png'.format(model_name, epoch)),im)

        # And a full pass over the validation data:
        val_batches = 0

        np.random.shuffle(filenames_val)
        val_gen = ParallelBatchIterator(load_images, filenames_val, ordered=True, batch_size=val_batch_size,multiprocess=False)

        for i, batch in enumerate(tqdm(val_gen)):
            inputs, targets = batch
            err, acc, l2_loss, true, prob, dice,tp,tn,fp,fn = val_fn(inputs, targets)

            val_metrics.append([err, acc, l2_loss, dice, tp, tn, fp, fn])
            val_batches += 1

            if np.ceil(i//val_batch_size) % 10 == 0: #Create image every 10th image
                im = np.hstack((
                    true[:OUTPUT_SIZE**2].reshape(OUTPUT_SIZE,OUTPUT_SIZE),
                    prob[:OUTPUT_SIZE**2].reshape(OUTPUT_SIZE,OUTPUT_SIZE)))

                plt.imsave(os.path.join(plot_folder,'val_{}_epoch{}.png'.format(model_name, epoch)),im)

        train_metrics = np.sum(np.array(train_metrics),axis=0)/train_batches
        val_metrics = np.sum(np.array(val_metrics),axis=0)/val_batches

        precision_train = train_metrics[4] / (train_metrics[4]+train_metrics[6])
        recall_train = train_metrics[4] / (train_metrics[4]+train_metrics[7])

        precision_val = val_metrics[4] / (val_metrics[4]+val_metrics[6])
        recall_val = val_metrics[4] / (val_metrics[4]+val_metrics[7])

        train_metrics = list(train_metrics[:5]) + [precision_train,recall_train] #Strip off false positives et al
        val_metrics = list(val_metrics[:5]) + [precision_val,recall_val]

        # Then we print the results for this epoch:
        print("\nEpoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        #print "Metrics"
        for name, train_metric, val_metric in zip(metric_names, train_metrics, val_metrics):
            print " {}:\t\t {:.6f}\t{:.6f}".format(name,train_metric,val_metric)

        if epoch % 4 == 0:
            print "Saving model"
            np.savez(os.path.join(model_folder,'{}_epoch{}.npz'.format(model_name, epoch)), *lasagne.layers.get_all_param_values(network))

        train_metrics_all.append(train_metrics)
        val_metrics_all.append(val_metrics)

        for name, train_vals, val_vals in zip(metric_names, zip(*train_metrics_all),zip(*val_metrics_all)):
            plt.figure()
            plt.plot(train_vals)
            plt.plot(val_vals)
            plt.ylabel(name)
            plt.xlabel(epoch)

            plt.savefig(os.path.join(plot_folder, '{}.png'.format(name)))


    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
