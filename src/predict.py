import sys
import params
from params import params as P
import numpy as np
import os

if __name__ == "__main__":
    model_folder = '../models/'

    if len(sys.argv) < 2:
        print "Missing arguments, first argument is model name, second is epoch"
        quit()

    model_folder = os.path.join(model_folder, sys.argv[1])
    P = params.Params(['../config/default.ini'] + [os.path.join(model_folder, 'config.ini')])


    import theano
    import theano.tensor as T
    import lasagne
    import unet
    from unet import INPUT_SIZE, OUTPUT_SIZE
    import dataset
    from parallel import ParallelBatchIterator
    from functools import partial


    from tqdm import tqdm
    from glob import glob

    input_var = T.tensor4('inputs')

    print "Defining network"
    net_dict = unet.define_network(input_var)
    network = net_dict['out']

    epoch = sys.argv[2]

    model_save_file = os.path.join(model_folder, P.MODEL_ID+"_epoch"+epoch+'.npz')

    print "Loading model save", model_save_file
    with np.load(model_save_file) as f:
         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    predict_fn = unet.define_predict(network, input_var)
