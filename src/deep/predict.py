from __future__ import division
import sys
import params
import numpy as np
import os
import skimage.io

model_folder = '../../models/'

if len(sys.argv) < 2:
    print "Missing arguments, first argument is model name, second is epoch"
    quit()

model_folder = os.path.join(model_folder, sys.argv[1])

#Overwrite params, ugly hack for now
params.params = params.Params(['../../config/default.ini'] + [os.path.join(model_folder, 'config.ini')])
from params import params as P
P.RANDOM_CROP = 0
P.INPUT_SIZE = 512
#P.INPUT_SIZE = 0


if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne
    sys.path.append('./unet')
    import unet
    import util
    from unet import INPUT_SIZE, OUTPUT_SIZE
    from dataset import load_images
    from parallel import ParallelBatchIterator
    from functools import partial
    from tqdm import tqdm
    from glob import glob


    input_var = T.tensor4('inputs')

    print "Defining network"
    net_dict = unet.define_network(input_var)
    network = net_dict['out']

    epoch = sys.argv[2]
    image_size = OUTPUT_SIZE**2

    model_save_file = os.path.join(model_folder, P.MODEL_ID+"_epoch"+epoch+'.npz')

    print "Loading saved model", model_save_file
    with np.load(model_save_file) as f:
         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    predict_fn = unet.define_predict(network, input_var)

    in_pattern = '../../data/1_1_1mm_slices_lung/subset[8-9]/*.pkl.gz'
    filenames = glob(in_pattern)#[:100]

    batch_size = 4
    multiprocess = False

    gen = ParallelBatchIterator(partial(load_images,deterministic=True),
                                        filenames, ordered=True,
                                        batch_size=batch_size,
                                        multiprocess=multiprocess)

    predictions_folder = os.path.join(model_folder, 'predictions_epoch{}'.format(epoch))
    util.make_dir_if_not_present(predictions_folder)

    print "Disabling warnings (saving empty images will warn user otherwise)"
    import warnings
    warnings.filterwarnings("ignore")

    for i, batch in enumerate(tqdm(gen)):
        inputs, _, weights, filenames = batch
        predictions = predict_fn(inputs)[0]
        #print inputs.shape, weights.shape
        for n, filename in enumerate(filenames):
            # Whole filepath without extension
            f = os.path.splitext(os.path.splitext(filename)[0])[0]

            # Filename only
            f = os.path.basename(f)
            f = os.path.join(predictions_folder,f+'.png')
            out_size = unet.output_size_for_input(inputs.shape[3], P.DEPTH)
            image_size = out_size**2
            image = predictions[n*image_size:(n+1)*image_size][:,1].reshape(out_size,out_size)

            #Remove parts outside a few pixels from the lungs
            image = image * np.where(weights[n,0,:,:]==0,0,1)

            image = np.array(np.round(image*255), dtype=np.uint8)

            skimage.io.imsave(f, image)
