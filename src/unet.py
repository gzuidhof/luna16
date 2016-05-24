import time
import numpy as np

import theano
import theano.tensor as T
import lasagne

from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, TransposedConv2DLayer
from lasagne.init import GlorotUniform
from lasagne import nonlinearities
from lasagne.layers import ConcatLayer
from lasagne.regularization import regularize_layer_params_weighted, l2, l1

from tqdm import tqdm


INPUT_SIZE = 572
OUTPUT_SIZE = 388

def define_network(input_var, target_var):
    batch_size = None
    net = {}

    net['input'] = InputLayer(shape=(batch_size,1,INPUT_SIZE,INPUT_SIZE), input_var=input_var)

    #First step
    net['conv1_1'] = Conv2DLayer(net['input'],
                                num_filters=64, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['conv1_2'] = Conv2DLayer(net['conv1_1'],
                                num_filters=64, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], pool_size=2, stride=2)

    # Second step
    net['conv2_1'] = Conv2DLayer(net['pool1'],
                                num_filters=128, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['conv2_2'] = Conv2DLayer(net['conv2_1'],
                                num_filters=128, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['pool2'] = MaxPool2DLayer(net['conv2_2'], pool_size=2, stride=2)

    # Third step
    net['conv3_1'] = Conv2DLayer(net['pool2'],
                                num_filters=256, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['conv3_2'] = Conv2DLayer(net['conv3_1'],
                                num_filters=256, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['pool3'] = MaxPool2DLayer(net['conv3_2'], pool_size=2, stride=2)

    # Fourth step
    net['conv4_1'] = Conv2DLayer(net['pool3'],
                                num_filters=512, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['conv4_2'] = Conv2DLayer(net['conv4_1'],
                                num_filters=512, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['pool4'] = MaxPool2DLayer(net['conv4_2'], pool_size=2, stride=2)

    # Last step
    net['conv5_1'] = Conv2DLayer(net['pool4'],
                                num_filters=1024, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)
    net['conv5_2'] = Conv2DLayer(net['conv5_1'],
                                num_filters=1024, filter_size=3, pad=0,
                                W=GlorotUniform(),
                                nonlinearity=nonlinearities.rectify)

    # Fourth unstep
    #net['unpool5'] = InverseLayer(net['conv5_2'], net['pool4'])
    net['upconv4'] = TransposedConv2DLayer(net['conv5_2'],
                                    num_filters=512, filter_size=2, stride=2,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['bridge4'] = ConcatLayer([net['upconv4']], axis=1, cropping=[None, None, 'center', 'center'])
    net['_conv4_2'] = Conv2DLayer(net['bridge4'], num_filters=512, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['_conv4_1'] = Conv2DLayer(net['_conv4_2'], num_filters=512, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

    # Third unstep
    net['upconv3'] = TransposedConv2DLayer(net['_conv4_1'],
                                    num_filters=256, filter_size=2, stride=2,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['bridge3'] = ConcatLayer([net['upconv3']], axis=1, cropping=[None, None, 'center', 'center'])
    net['_conv3_2'] = Conv2DLayer(net['bridge3'], num_filters=256, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['_conv3_1'] = Conv2DLayer(net['_conv3_2'], num_filters=256, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

    # Second unstep
    net['upconv2'] = TransposedConv2DLayer(net['_conv3_1'],
                                    num_filters=128, filter_size=2, stride=2,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['bridge2'] = ConcatLayer([net['upconv2']], axis=1, cropping=[None, None, 'center', 'center'])
    net['_conv2_2'] = Conv2DLayer(net['bridge2'], num_filters=128, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['_conv2_1'] = Conv2DLayer(net['_conv2_2'], num_filters=128, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

    # First unstep
    net['upconv1'] = TransposedConv2DLayer(net['_conv2_1'],
                                    num_filters=64, filter_size=2, stride=2,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['bridge1'] = ConcatLayer([net['upconv1']], axis=1, cropping=[None, None, 'center', 'center'])
    net['_conv1_2'] = Conv2DLayer(net['bridge1'], num_filters=64, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)
    net['_conv1_1'] = Conv2DLayer(net['_conv1_2'], num_filters=64, filter_size=3, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

    # Output layer
    net['out'] = Conv2DLayer(net['_conv1_1'], num_filters=1, filter_size=(1,1), pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

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


if __name__ == "__main__":
    # create Theano variables for input and target minibatch
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')#T.ivector('targets')

    print "Defining network"
    net_dict = define_network(input_var, target_var)
    network = net_dict['out']

    params = lasagne.layers.get_all_params(network, trainable=True)
    loss_weighing = target_var*3+1


    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss * loss_weighing
    loss = loss.mean()

    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.99)


    test_prediction = lasagne.layers.get_output(network, deterministic=True)


    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss * loss_weighing
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    print "Defining train function"
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    print "Defining validation function"
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])



    X_train = np.array(np.random.rand(10,1,INPUT_SIZE,INPUT_SIZE), dtype=np.float32)
    y_train = np.array(np.random.randint(2,size=(10,1,OUTPUT_SIZE,OUTPUT_SIZE)), dtype=np.float32)

    X_val = np.array(np.random.rand(10,1,INPUT_SIZE,INPUT_SIZE), dtype=np.float32)
    y_val = np.array(np.random.randint(2,size=(10,1,OUTPUT_SIZE,OUTPUT_SIZE)), dtype=np.float32)

    X_test = np.array(np.random.rand(10,1,INPUT_SIZE,INPUT_SIZE), dtype=np.float32)
    y_test = np.array(np.random.randint(2,size=(10,1,OUTPUT_SIZE,OUTPUT_SIZE)), dtype=np.float32)

    num_epochs = 100

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 1, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            #print "batcheroo"

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 2, shuffle=False):
            inputs, targets = batch
            print inputs.shape
            print targets.shape
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 1, shuffle=False):
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
