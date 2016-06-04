import sys
import params
from params import params as P
import numpy as np
import os
import skimage.io

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

    in_pattern = '../data/1_1_1mm_512_x_512_lung_slices/subset9/*.pkl.gz'
    filenames = glob(in_pattern)[:100]

    batch_size = P.BATCH_SIZE_VALIDATION
    multiprocess = False

    gen = ParallelBatchIterator(partial(load_images,deterministic=True),
                                        filenames, ordered=True,
                                        batch_size=batch_size,
                                        multiprocess=multiprocess)

    predictions_folder = os.path.join(model_folder, 'predictions_epoch{}'.format(epoch))
    util.make_dir_if_not_present(predictions_folder)

    for i, batch in enumerate(tqdm(gen)):
        inputs, _, _, filenames = batch
        predictions = predict_fn(inputs)[0]

        for n, filename in enumerate(filenames):
            # Whole filepath without extension
            f = os.path.splitext(os.path.splitext(filename)[0])[0]

            # Filename only
            f = os.path.basename(f)
            f = os.path.join(predictions_folder,f+'.png')

            image = predictions[n*image_size:(n+1)*image_size][:,1].reshape(OUTPUT_SIZE,OUTPUT_SIZE)
            image = np.array(image*255, dtype=np.uint8)

            skimage.io.imsave(f, image)
