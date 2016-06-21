import numpy as np
import matplotlib.pyplot as plt
import time
import theano
import theano.tensor as T
import lasagne
import sys
sys.path.append('../')
from lasagne.regularization import regularize_layer_params, l2

from math import sqrt, ceil
import os
from tqdm import tqdm

from params import params
import dataset_3D
import normalize
from lasagne.init import HeNormal
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer


def define_network(inputs):

    network = lasagne.layers.InputLayer(shape=(None, params.CHANNELS, params.INPUT_SIZE, params.INPUT_SIZE, params.INPUT_SIZE),
                                input_var=inputs)

    network = Conv3DDNNLayer(
            network, num_filters=64, filter_size=(5, 5, 5),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=HeNormal(gain='relu'))

    network = MaxPool3DDNNLayer(network, pool_size=(2, 2, 2))

    if params.BATCH_NORMALIZATION:
        network = lasagne.layers.batch_norm(network)

    network = Conv3DDNNLayer(
            network, num_filters=64, filter_size=(5, 5, 5),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=HeNormal(gain='relu'))

    network = Conv3DDNNLayer(
            network, num_filters=96, filter_size=(5, 5, 5),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=HeNormal(gain='relu'))

    if params.BATCH_NORMALIZATION:
        network = lasagne.layers.batch_norm(network)

    network = lasagne.layers.DenseLayer(
            network,
            num_units=420,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=HeNormal(gain='relu')
    )

    network = lasagne.layers.DenseLayer(
            network, num_units=params.N_CLASSES,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def define_updates(network, inputs, targets):
    prediction = lasagne.layers.get_output(network)

    loss = lasagne.objectives.categorical_crossentropy(T.clip(prediction, 0.00001, 0.99999), targets)
    loss = loss.mean()

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(T.clip(test_prediction, 0.00001, 0.99999), targets)
    test_loss = test_loss.mean()


    l2_loss = regularize_layer_params(network, l2) * params.L2_LAMBDA
    loss = loss + l2_loss
    test_loss = test_loss + l2_loss


    acc = T.mean(T.eq(T.argmax(prediction, axis=1), targets),
                dtype=theano.config.floatX)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), targets),
                dtype=theano.config.floatX)


    l_r = theano.shared(np.array(params.LEARNING_RATE, dtype=theano.config.floatX))

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD), but Lasagne offers plenty more.
    network_params = lasagne.layers.get_all_params(network, trainable=True)
    if params.OPTIMIZATION == "MOMENTUM":
        updates = lasagne.updates.momentum(loss, network_params, learning_rate=l_r, momentum=params.MOMENTUM)
    elif params.OPTIMIZATION == "ADAM":
        updates = lasagne.updates.adam(loss, network_params, learning_rate=l_r)
    elif params.OPTIMIZATION == "RMSPROP":
        updates = lasagne.updates.adam(loss, network_params)

    prediction_binary = T.argmax(prediction, axis=1)
    test_prediction_binary = T.argmax(test_prediction, axis=1)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([inputs, targets], [loss, l2_loss, acc, prediction_binary], updates=updates)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([inputs, targets], [test_loss, l2_loss, test_acc, test_prediction_binary])

    return train_fn, val_fn, l_r
