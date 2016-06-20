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

from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer


def define_network(inputs):

    network = lasagne.layers.InputLayer(shape=(None, params.CHANNELS, params.PIXELS, params.PIXELS),
                                input_var=inputs)

    network = Conv3DDNNLayer(
            network, num_filters=64, filter_size=(5, 5, 3),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())

    network = MaxPool3DDNNLayer(network, pool_size=(2, 2, 1))

    if params.BATCH_NORMALIZATION:
        network = lasagne.layers.BatchNormLayer(network)

    network = Conv3DDNNLayer(
            network, num_filters=64, filter_size=(5, 5, 3),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())

    network = Conv3DDNNLayer(
            network, num_filters=64, filter_size=(5, 5, 3),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())

    if params.BATCH_NORMALIZATION:
        network = lasagne.layers.BatchNormLayer(network)

    network = lasagne.layers.DenseLayer(
            network,
            num_units=250,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform()
    )

    network = lasagne.layers.DenseLayer(
            network, num_units=params.N_CLASSES,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def define_loss(network, targets):
    prediction = lasagne.layers.get_output(network)


    loss = lasagne.objectives.categorical_crossentropy(prediction, targets)
    loss = loss.mean()

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, targets)
    test_loss = test_loss.mean()


    regularization_penalty = regularize_layer_params(network, l2) * params.L2_LAMBDA

    loss = loss + regularization_penalty
    test_loss = test_loss + regularization_penalty

    acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), targets),
                dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([inputs, targets], [test_prediction, test_loss, acc])

    return loss, val_fn


def define_learning(network, loss):
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD), but Lasagne offers plenty more.
    network_params = lasagne.layers.get_all_params(network, trainable=True)
    if params.OPTIMIZATION == "MOMENTUM":
        updates = lasagne.updates.momentum(loss, network_params, learning_rate=params.LEARNING_RATE, momentum=params.MOMENTUM)
    elif params.OPTIMIZATION == "ADAM":
        updates = lasagne.updates.adam(loss, network_params, learning_rate=params.LEARNING_RATE)
    elif params.OPTIMIZATION == "RMSPROP":
        updates = lasagne.updates.adam(loss, network_params)



    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([inputs, targets], loss, updates=updates)

    return train_fn