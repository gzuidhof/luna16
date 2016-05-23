import numpy as np

import theano
import theano.tensor as T

from lasagne.layers import (InputLayer, Conv2DLayer,
MaxPool2DLayer, TransposedConv2DLayer)
from lasagne.init import GlorotUniform
from lasagne import nonlinearities
from lasagne.layers import ConcatLayer


def define_network(input_var, target_var):
    batch_size = 1
    net = {}

    net['input'] = InputLayer(shape=(batch_size,1,572,572), input_var=input_var)

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
    net['out'] = Conv2DLayer(net['_conv1_1'], num_filters=1, filter_size=1, pad=0,
                                    W=GlorotUniform(),
                                    nonlinearity=nonlinearities.rectify)

    return net

if __name__ == "__main__":
    # create Theano variables for input and target minibatch
    input_var = T.tensor4('X')
    target_var = T.ivector('y')
    net_dict = define_network(input_var, target_var)
    network = net_dict['out']
