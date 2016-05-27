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

    from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, InverseLayer

    from lasagne.init import GlorotUniform, HeNormal
    from lasagne import nonlinearities
    from lasagne.layers import ConcatLayer, Upscale2DLayer
    from lasagne.regularization import l2, l1, regularize_network_params

from tqdm import tqdm
from glob import glob
import gzip

import cPickle as pickle
from parallel import ParallelBatchIterator

#INPUT_SIZE = 572
INPUT_SIZE = 512
NET_DEPTH = 5

SIZE_DICT = {
    2:556,
    3:532,
    4:484,
    5:388
}

SIZE_DICT_512 = {
    5:324
}


OUTPUT_SIZE = SIZE_DICT_512[NET_DEPTH]

def filter_for_depth(depth):
    return 2**(5+depth)

folder = '../data/train_resized/'

def define_network(input_var, target_var):
    batch_size = None
    net = {}
    net['input'] = InputLayer(shape=(batch_size,1,INPUT_SIZE,INPUT_SIZE), input_var=input_var)

    nonlinearity = nonlinearities.rectify



    def contraction(depth, pool=True):
        n_filters = filter_for_depth(depth)
        incoming = net['input'] if depth == 0 else net['pool{}'.format(depth-1)]

        net['conv{}_1'.format(depth)] = Conv2DLayer(incoming,
                                    num_filters=n_filters, filter_size=3, pad='valid',
                                    W=HeNormal(gain='relu'),
                                    nonlinearity=nonlinearity)
        net['conv{}_2'.format(depth)] = Conv2DLayer(net['conv{}_1'.format(depth)],
                                    num_filters=n_filters, filter_size=3, pad='valid',
                                    W=HeNormal(gain='relu'),
                                    nonlinearity=nonlinearity)

        if pool:
            net['pool{}'.format(depth)] = MaxPool2DLayer(net['conv{}_2'.format(depth)], pool_size=2, stride=2)

    def expansion(depth):
        n_filters = filter_for_depth(depth)

        deepest = 'pool{}'.format(depth+1) not in net
        incoming = net['conv{}_2'.format(depth+1)] if deepest else net['_conv{}_2'.format(depth+1)]

        #net['upconv{}'.format(depth)] = TransposedConv2DLayer(incoming,
        #                                num_filters=n_filters, filter_size=2, stride=2,
        #                                W=HeNormal(gain='relu'),
        #                                nonlinearity=nonlinearities.rectify)
        upsc = Upscale2DLayer(incoming,4)
        net['upconv{}'.format(depth)] = Conv2DLayer(upsc,
                                        num_filters=n_filters, filter_size=2, stride=2,
                                        W=HeNormal(gain='relu'),
                                        nonlinearity=nonlinearity)


        net['bridge{}'.format(depth)] = ConcatLayer([net['upconv{}'.format(depth)],net['conv{}_2'.format(depth)]], axis=1, cropping=[None, None, 'center', 'center'])
        net['_conv{}_1'.format(depth)] = Conv2DLayer(net['bridge{}'.format(depth)],
                                        num_filters=n_filters, filter_size=3, pad='valid',
                                        W=HeNormal(gain='relu'),
                                        nonlinearity=nonlinearity)
        net['_conv{}_2'.format(depth)] = Conv2DLayer(net['_conv{}_1'.format(depth)],
                                        num_filters=n_filters, filter_size=3, pad='valid',
                                        W=HeNormal(gain='relu'),
                                        nonlinearity=nonlinearity)


    for d in range(NET_DEPTH):
        is_not_deepest = d!=NET_DEPTH-1
        contraction(d, pool=is_not_deepest)

    for d in reversed(range(NET_DEPTH-1)):
        expansion(d)

    # Output layer
    net['out'] = Conv2DLayer(net['_conv0_2'], num_filters=2, filter_size=(1,1), pad='valid',
                                    nonlinearity=None)

    #import network_repr
    #print network_repr.get_network_str(net['out'])
    print lasagne.layers.get_output_shape(net['out'])

    return net

def dice(train,truth):
    score = np.sum(train[truth>0])*2.0 / (np.sum(train) + np.sum(truth))
    return score


if __name__ == "__main__":
    l2_lambda = 1e-6

    # create Theano variables for input and target minibatch
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets', dtype='int64')

    print "Defining network"
    net_dict = define_network(input_var, target_var)
    network = net_dict['out']

    params = lasagne.layers.get_all_params(network, trainable=True)
    target_prediction = target_var.dimshuffle(1,0,2,3).flatten(ndim=1)

    _EPSILON=1e-8
    true_case_weight = (1/(T.mean(target_prediction)+_EPSILON))#*0.8
    #true_case_weight=1.5
    loss_weighing = (true_case_weight-1)*target_prediction + 1

    prediction = lasagne.layers.get_output(network)
    prediction_flat = prediction.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)

    #theano.printing.debugprint(prediction)

    softmax = lasagne.nonlinearities.softmax(prediction_flat)
    prediction_binary = T.argmax(softmax, axis=1)

    dice_score = T.sum(T.eq(2, prediction_binary+target_prediction))*2.0 / (T.sum(prediction_binary) + T.sum(target_prediction))
    l2_loss = l2_lambda * regularize_network_params(network, l2)

    loss = lasagne.objectives.categorical_crossentropy(T.clip(softmax,_EPSILON,1-_EPSILON), target_prediction)
    loss = loss * loss_weighing
    loss = loss.mean()
    #loss += (1-dice_score)**2
    loss += l2_loss


    acc = T.mean(T.eq(prediction_binary, target_prediction),
                      dtype=theano.config.floatX)

    updates = lasagne.updates.nesterov_momentum(
            #loss, params, learning_rate=0.00001, momentum=0.9)
            loss, params, learning_rate=0.00002, momentum=0.99)


    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction_flat = test_prediction.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
    test_softmax = lasagne.nonlinearities.softmax(prediction_flat)

    test_loss = lasagne.objectives.categorical_crossentropy(T.clip(test_softmax,_EPSILON,1-_EPSILON), target_prediction)
    #test_loss = test_loss * loss_weighing
    test_loss = test_loss.mean()
    test_loss += l2_loss

    test_acc = T.mean(T.eq(T.argmax(test_softmax, axis=1), target_prediction),
                      dtype=theano.config.floatX)

    print "Defining train function"
    train_fn = theano.function([input_var, target_var],[loss, acc, l2_loss, prediction_binary, target_prediction, softmax[:,1], dice_score], updates=updates)

    print "Defining validation function"
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    #filenames = [f for f in glob(folder+ '*.tif') if 'mask' not in f]
    folder = './../data/1_1_1mm_512_x_512_lung_slices/subset0/'
    filenames = glob(folder+ '*.pkl.gz')
    #print filenames
    np.random.seed(1)
    np.random.shuffle(filenames)
    filenames_train = filenames[:150]
    filenames_val = filenames[600:700]
    filenames_test = filenames_val

    #filenames_train = filenames_train[2:3]
    #print filenames_train
    filenames_val = filenames_val[:200]
    filenames_test = filenames_val[:200]

    num_epochs = 400

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        train_l2 = 0
        train_dice = 0
        start_time = time.time()

        np.random.shuffle(filenames_train)
        train_gen = ParallelBatchIterator(load_images, filenames_train, ordered=True, batch_size=1, multiprocess=False)

        #for batch in iterate_minibatches(X_train, y_train, 1, shuffle=True):
        for i, batch in enumerate(tqdm(train_gen)):
            inputs, targets = batch
            err, acc, l2_loss, pred, true, prob, d = train_fn(inputs, targets)


            if i % 10 == 0:
                im = np.hstack((
                    true[:OUTPUT_SIZE**2].reshape(OUTPUT_SIZE,OUTPUT_SIZE),
                    prob[:OUTPUT_SIZE**2].reshape(OUTPUT_SIZE,OUTPUT_SIZE)))

                plt.imsave('../images/plot/epoch{}.png'.format(epoch),im)
            #plt.show()
            train_err += err
            train_acc += acc
            train_batches += 1
            train_l2 += l2_loss
            train_dice += d
            #print "batcheroo"

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0

        val_gen = ParallelBatchIterator(load_images, filenames_val, ordered=True, batch_size=2,multiprocess=False)

        for batch in tqdm(val_gen):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  l2 loss:\t\t\t{:.6f}".format(l2_loss/train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))
        print("  training dice:\t\t{:.5f}".format(
            train_dice / train_batches))


        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    test_gen = ParallelBatchIterator(load_images, filenames_test, ordered=True, batch_size=2)
    #for batch in iterate_minibatches(X_test, y_test, 3, shuffle=False):
    for batch in test_gen:
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
