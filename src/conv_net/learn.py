import numpy as np
import matplotlib.pyplot as plt
import time
import theano
import theano.tensor as T
import lasagne

from lasagne.regularization import regularize_layer_params, l2

from math import sqrt, ceil
import os
from tqdm import tqdm

from params import params
import data
import normalize
from augment import Augmenter
from util import iterate_minibatches, histogram_equalization


def define_network(inputs):

    network = lasagne.layers.InputLayer(shape=(None, params.CHANNELS, params.PIXELS, params.PIXELS),
                                input_var=inputs)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    if params.BATCH_NORMALIZATION:
        network = lasagne.layers.BatchNormLayer(network)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=96, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    if params.BATCH_NORMALIZATION:
        network = lasagne.layers.BatchNormLayer(network)

    if params.DEEPER:
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=128, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                W=lasagne.init.GlorotUniform())
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=128, filter_size=(1, 1),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                W=lasagne.init.GlorotUniform())

        #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        if params.BATCH_NORMALIZATION:
            network = lasagne.layers.BatchNormLayer(network)

    if params.EVEN_DEEPER:
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=128, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                W=lasagne.init.GlorotUniform())

        #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        if params.BATCH_NORMALIZATION:
            network = lasagne.layers.BatchNormLayer(network)

    network = lasagne.layers.DenseLayer(
            network,
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform()
    )
    network = lasagne.layers.DenseLayer(
            network,
            num_units=1024,
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

    if params.REGULARIZATION:
        regularization_penalty = regularize_layer_params(network, l2) * params.REGULARIZATION_WEIGHT

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
    if params.UPDATEFUNCTION == "MOMENTUM":
        updates = lasagne.updates.momentum(loss, network_params, learning_rate=params.START_LEARNING_RATE, momentum=params.MOMENTUM)
    elif params.UPDATEFUNCTION == "ADAM":
        updates = lasagne.updates.adam(loss, network_params, learning_rate=params.START_LEARNING_RATE)
    elif params.UPDATEFUNCTION == "RMSPROP":
        updates = lasagne.updates.adam(loss, network_params)



    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([inputs, targets], loss, updates=updates)

    return train_fn

if __name__ == "__main__":
    np.random.seed(0)

    # First we define the symbolic input X and the symbolic target y. We want
    # to solve the equation y = C(X) where C is a classifier (convolutional network).
    inputs = T.tensor4('X')
    targets = T.ivector('y')

    print "Defining network"
    network = define_network(inputs)
    print "Defining loss function"
    loss, val_fn = define_loss(network, targets)
    print "Defining learning function"
    train_fn = define_learning(network, loss)

    print "Loading data"
    train_X, train_y, val_X, val_y, test_X, test_y, label_to_names = data.load()

    print "Determining mean and std of train set"
    mean, std = normalize.calc_mean_std(train_X)


    a = Augmenter(multiprocess=True)

    # The number of epochs specifies the number of passes over the whole training data
    # Depending on augmentation settings, it still improves through epoch 100..
    num_epochs = 100

    #Take subset? Speeds it up x2, but worse performance ofc
    #train_X = train_X[:20000]
    #train_y = train_y[:20000]

    if params.HISTOGRAM_EQUALIZATION:
        adaptive = params.CLAHE
        print "Performing equalization (adaptive={})".format(adaptive)
        train_X = histogram_equalization(train_X, adaptive)
        val_X = histogram_equalization(val_X, adaptive)
        test_X = histogram_equalization(test_X, adaptive)



    print "Training for {} epochs".format(num_epochs)

    curves = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data...
        train_err = 0
        train_batches = 0
        start_time = time.time()


        aug_time = 0
        for batch in tqdm(iterate_minibatches(train_X, train_y, 256, shuffle=True)):
            inputs, targets = batch
            if params.AUGMENT:
                pre_aug = time.time()
                inputs_augmented = a.augment(inputs)
                aug_time+= (time.time() - pre_aug)

                #Show unaugmented and augmented images
                #visualize_data(np.append(inputs[:8],inputs_augmented[:8],axis=0).transpose(0,2,3,1))

                inputs_augmented = normalize.normalize(inputs_augmented, mean, std)
                train_err += train_fn(inputs_augmented, targets)
            else:
                #print inputs.shape,targets.shape
                train_err += train_fn(inputs, targets)
            train_batches += 1

        #print "Augmentation time: ", aug_time
        # ...and a full pass over the validation data
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(val_X, val_y, 500, shuffle=False):
            inputs, targets = batch

            inputs = normalize.normalize(inputs, mean, std)

            preds, err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s ({:.3f}s augmentation)".format(
            epoch + 1, num_epochs, time.time() - start_time, aug_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        curves['train_loss'].append(train_err / train_batches)
        curves['val_loss'].append(val_err / val_batches)
        curves['val_acc'].append(val_acc / val_batches)

    print "Predicting test set"
    test_err = 0
    test_acc = 0
    test_batches = 0

    for batch in iterate_minibatches(test_X, test_y, 500, shuffle=False):
        inputs, targets = batch

        inputs = normalize.normalize(inputs, mean, std)

        preds, err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    print("TEST loss:\t\t{:.6f}".format(test_err / test_batches))
    print("TEST accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

    print "Plotting"
    plt.plot(zip(curves['train_loss'], curves['val_loss']));
    plt.plot(curves['val_acc']);
    plt.show()
