from __future__ import division
import time
import numpy as np
import normalize

if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne

    from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, TransposedConv2DLayer, NonlinearityLayer
    from lasagne.init import GlorotUniform
    from lasagne import nonlinearities
    from lasagne.layers import ConcatLayer
    from lasagne.regularization import regularize_layer_params_weighted, l2, l1

    from lasagne.layers import autocrop

from tqdm import tqdm
from glob import glob
import gzip

import cPickle as pickle
from parallel import ParallelBatchIterator



INPUT_SIZE = 572
OUTPUT_SIZE = 388

FILTER_STEP1 = 16
FILTER_STEP2 = 32
FILTER_STEP3 = 64
FILTER_STEP4 = 128
FILTER_STEP5 = 256

def define_network(input_var, target_var):
    batch_size = None
    net = {}

    net['input'] = InputLayer(shape=(batch_size,1,INPUT_SIZE,INPUT_SIZE), input_var=input_var)

    #First step
    net['conv1_1'] = Conv2DLayer(net['input'],
                                num_filters=FILTER_STEP1, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['conv1_2'] = Conv2DLayer(net['conv1_1'],
                                num_filters=FILTER_STEP1, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], pool_size=2, stride=2)

    # Second step
    net['conv2_1'] = Conv2DLayer(net['pool1'],
                                num_filters=FILTER_STEP2, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['conv2_2'] = Conv2DLayer(net['conv2_1'],
                                num_filters=FILTER_STEP2, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['pool2'] = MaxPool2DLayer(net['conv2_2'], pool_size=2, stride=2)

    # Third step
    net['conv3_1'] = Conv2DLayer(net['pool2'],
                                num_filters=FILTER_STEP3, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['conv3_2'] = Conv2DLayer(net['conv3_1'],
                                num_filters=FILTER_STEP3, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['pool3'] = MaxPool2DLayer(net['conv3_2'], pool_size=2, stride=2)

    # Fourth step
    net['conv4_1'] = Conv2DLayer(net['pool3'],
                                num_filters=FILTER_STEP4, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['conv4_2'] = Conv2DLayer(net['conv4_1'],
                                num_filters=FILTER_STEP4, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['pool4'] = MaxPool2DLayer(net['conv4_2'], pool_size=2, stride=2)

    # Last step
    net['conv5_1'] = Conv2DLayer(net['pool4'],
                                num_filters=FILTER_STEP5, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['conv5_2'] = Conv2DLayer(net['conv5_1'],
                                num_filters=FILTER_STEP5, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)

    # Fourth unstep
    #net['unpool5'] = InverseLayer(net['conv5_2'], net['pool4'])
    net['upconv4'] = TransposedConv2DLayer(net['conv5_2'],
                                    num_filters=FILTER_STEP4, filter_size=2, stride=2,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['bridge4'] = ConcatLayer([net['upconv4']], axis=1, cropping=[None, None, 'center', 'center'])
    net['_conv4_2'] = Conv2DLayer(net['bridge4'], num_filters=FILTER_STEP4, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['_conv4_1'] = Conv2DLayer(net['_conv4_2'], num_filters=FILTER_STEP4, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

    # Third unstep
    net['upconv3'] = TransposedConv2DLayer(net['_conv4_1'],
                                    num_filters=FILTER_STEP3, filter_size=2, stride=2,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['bridge3'] = ConcatLayer([net['upconv3']], axis=1, cropping=[None, None, 'center', 'center'])
    net['_conv3_2'] = Conv2DLayer(net['bridge3'], num_filters=FILTER_STEP3, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['_conv3_1'] = Conv2DLayer(net['_conv3_2'], num_filters=FILTER_STEP3, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

    # Second unstep
    net['upconv2'] = TransposedConv2DLayer(net['_conv3_1'],
                                    num_filters=FILTER_STEP2, filter_size=2, stride=2,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['bridge2'] = ConcatLayer([net['upconv2']], axis=1, cropping=[None, None, 'center', 'center'])
    net['_conv2_2'] = Conv2DLayer(net['bridge2'], num_filters=FILTER_STEP2, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['_conv2_1'] = Conv2DLayer(net['_conv2_2'], num_filters=FILTER_STEP2, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

    # First unstep
    net['upconv1'] = TransposedConv2DLayer(net['_conv2_1'],
                                    num_filters=64, filter_size=2, stride=2,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['bridge1'] = ConcatLayer([net['upconv1']], axis=1, cropping=[None, None, 'center', 'center'])
    net['_conv1_2'] = Conv2DLayer(net['bridge1'], num_filters=FILTER_STEP1, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['_conv1_1'] = Conv2DLayer(net['_conv1_2'], num_filters=FILTER_STEP1, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

    # Output layer
    net['out'] = Conv2DLayer(net['_conv1_1'], num_filters=1, filter_size=(1,1), pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

    #net['out'] = NonlinearityLayer(net['flap'], nonlinearity=lasagne.nonlinearities.softmax)
    #print lasagne.layers.get_output_shape(net['out'])
    #print lasagne.layers.get_output_shape(net['out'])
    return net

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

x_folder = '../data/1_1_1mm_512_x_512_lung_slices/subset0/'

def load_slice(filename):
    #print "----------------"+str(filename)+"----------------"
    with gzip.open(filename,'rb') as f:
        lung = pickle.load(f)

    with gzip.open(filename.replace('lung','nodule'),'rb') as f:
        truth = pickle.load(f)

    lung = np.pad(lung, (INPUT_SIZE-lung.shape[0])//2, 'constant', constant_values=-400)

    lung = np.array(normalize.normalize(lung),dtype=np.float32)

    # Crop truth
    crop_size = OUTPUT_SIZE
    offset = (truth.shape[0]-crop_size)//2
    truth = truth[offset:offset+crop_size,offset:offset+crop_size]

    lung = np.expand_dims(np.expand_dims(lung, axis=0),axis=0)
    truth = np.array(np.expand_dims(np.expand_dims(truth, axis=0),axis=0),dtype=np.float32)

    return lung, truth

def load_slice_multiple(filenames):
    slices = map(load_slice, filenames)
    lungs, truths = zip(*slices)

    #print np.concatenate(lungs,axis=0).shape

    return np.concatenate(lungs,axis=0), np.concatenate(truths,axis=0)


if __name__ == "__main__":
    # create Theano variables for input and target minibatch
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')#T.ivector('targets')

    print "Defining network"
    net_dict = define_network(input_var, target_var)
    network = net_dict['out']

    params = lasagne.layers.get_all_params(network, trainable=True)
    loss_weighing = target_var*10+1


    prediction = lasagne.layers.get_output(network)
    #loss = lasagne.objectives.binary_crossentropy(T.clip(prediction,0.001,0.999), target_var)
    #e = T.exp(prediction)
    #softmax = e[:,0,:,:]
    #softmax = (e / T.stack([e.sum(axis=1),e.sum(axis=1)], axis=1))[:,0,:,:]
    #print softmax.shape

    #loss = lasagne.objectives.binary_crossentropy(softmax, target_var)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss * loss_weighing
    loss = loss.mean()


    #acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
    #                  dtype=theano.config.floatX)
    acc = T.mean(T.eq(T.gt(prediction, 0.5), target_var),
                      dtype=theano.config.floatX)

    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0001, momentum=0.99)


    test_prediction = lasagne.layers.get_output(network, deterministic=True)


    #e_test = T.exp(test_prediction)
    #softmax_test = (e_test / T.stack([e_test.sum(axis=1),e_test.sum(axis=1)], axis=1))

    #test_loss = lasagne.objectives.binary_crossentropy(softmax_test, target_var)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    #test_loss = lasagne.objectives.binary_crossentropy(T.clip(test_prediction,0.001,0.999), target_var)
    test_loss = test_loss * loss_weighing
    test_loss = test_loss.mean()
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                  dtype=theano.config.floatX)
    test_acc = T.mean(T.eq(T.gt(test_prediction, 0.5), target_var),
                      dtype=theano.config.floatX)


    print "Defining train function"
    train_fn = theano.function([input_var, target_var],[loss, acc], updates=updates)

    print "Defining validation function"
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


    filenames = glob(x_folder+ '*.pkl.gz')
    np.random.shuffle(filenames)
    filenames_train = filenames[:600]
    filenames_val = filenames[600:]
    filenames_test = filenames_val

    filenames_train = filenames_train[:500]
    filenames_val = filenames_val[:100]
    filenames_test = filenames_test[100:200]

    num_epochs = 400

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()

        np.random.shuffle(filenames_train)
        #filenames_train = shuffle(filenames_train)
        #train_gen = ParallelBatchIterator(load_slice, filenames_train, ordered=True, batch_size=1)
        train_gen = ParallelBatchIterator(load_slice_multiple, filenames_train, ordered=True, batch_size=4)

        #for batch in iterate_minibatches(X_train, y_train, 1, shuffle=True):
        for batch in tqdm(train_gen):
            inputs, targets = batch
            #print inputs.shape
            #print targets.shape
            err, acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1
            #print "batcheroo"

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0

        #val_gen = ParallelBatchIterator(load_slice_multiple, filenames_val, ordered=True, batch_size=2)
        val_gen = ParallelBatchIterator(load_slice, filenames_val, ordered=True, batch_size=1)

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
        print("  training accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))

        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    test_gen = ParallelBatchIterator(load_slice, filenames_test, ordered=True, batch_size=2)
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
