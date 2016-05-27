import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.init import HeNormal
from lasagne import nonlinearities
from lasagne.layers import ConcatLayer, Upscale2DLayer
from lasagne.regularization import l2, regularize_network_params

#INPUT_SIZE = 572 # The standard size
INPUT_SIZE = 512 #Reduced size, also fits lungs easily in output map
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

def define_network(input_var):
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
    print 'Network output shape', lasagne.layers.get_output_shape(net['out'])
    return net

def define_updates(network, input_var, target_var):
    l2_lambda = 1e-5 #Weight decay
    learning_rate = learning_rate=0.00002
    momentum = 0.99

    params = lasagne.layers.get_all_params(network, trainable=True)
    target_prediction = target_var.dimshuffle(1,0,2,3).flatten(ndim=1)

    _EPSILON=1e-8
    true_case_weight = (1/(T.mean(target_prediction)+_EPSILON))#*0.8
    loss_weighing = (true_case_weight-1)*target_prediction + 1

    prediction = lasagne.layers.get_output(network)
    prediction_flat = prediction.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)

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

    # Can be used for precision/recall
    tp = (T.eq(target_prediction,1) * T.eq(prediction_binary,1)).sum()
    tn = (T.neq(target_prediction,1) * T.neq(prediction_binary,1)).sum()
    fp = (T.neq(target_prediction,1) * T.eq(prediction_binary,1)).sum()
    fn = (T.eq(target_prediction,1) * T.neq(prediction_binary,1)).sum()

    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=momentum)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction_flat = test_prediction.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
    test_softmax = lasagne.nonlinearities.softmax(prediction_flat)

    test_loss = lasagne.objectives.categorical_crossentropy(T.clip(test_softmax,_EPSILON,1-_EPSILON), target_prediction)
    test_loss = test_loss * loss_weighing
    test_loss = test_loss.mean()
    test_loss += l2_loss

    test_prediction_binary = T.argmax(softmax, axis=1)
    test_dice_score = T.sum(T.eq(2, test_prediction_binary+target_prediction))*2.0 / (T.sum(test_prediction_binary) + T.sum(target_prediction))

    test_acc = T.mean(T.eq(test_prediction_binary, target_prediction),
                      dtype=theano.config.floatX)

    # Can be used for precision/recall
    t_tp = (T.eq(target_prediction,1) * T.eq(test_prediction_binary,1)).sum()
    t_tn = (T.neq(target_prediction,1) * T.neq(test_prediction_binary,1)).sum()
    t_fp = (T.neq(target_prediction,1) * T.eq(test_prediction_binary,1)).sum()
    t_fn = (T.eq(target_prediction,1) * T.neq(test_prediction_binary,1)).sum()

    print "Defining train function"
    train_fn = theano.function([input_var, target_var],[
                                loss, acc, l2_loss, target_prediction, softmax[:,1],
                                dice_score, tp,tn,fp,fn],
                                updates=updates)

    print "Defining validation function"
    val_fn = theano.function([input_var, target_var], [
                                test_loss, test_acc, l2_loss, target_prediction, test_softmax[:,1],
                                test_dice_score, t_tp, t_tn, t_fp, t_fn])

    return train_fn, val_fn
