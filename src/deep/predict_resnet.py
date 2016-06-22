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

#Run with: python predict_resnet.py 1466485849_resnet 194 9

#Overwrite params, ugly hack for now
params.params = params.Params(['../../config/default.ini'] + [os.path.join(model_folder, 'config.ini')])
from params import params as P

if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne
    sys.path.append('./resnet')
    import util
    from dataset_2D import load_images
    from parallel import ParallelBatchIterator
    from functools import partial
    from tqdm import tqdm
    from glob import glob
    import resnet
    import pandas as pd
    print "Defining network"

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = resnet.ResNet_FullPre_Wide(input_var, P.DEPTH, P.BRANCHING_FACTOR)

    epoch = sys.argv[2]
    subsets = sys.argv[3]

    model_save_file = os.path.join(model_folder, P.MODEL_ID+"_epoch"+epoch+'.npz')

    print "Loading saved model", model_save_file
    with np.load(model_save_file) as f:
         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    train_fn, val_fn, l_r = resnet.define_updates(network, input_var, target_var)

    in_pattern = '../../data/cadV2_0.5mm_64x64_xy_xz_yz/subset[{}]/*/*.pkl.gz'.format(subsets)
    filenames = glob(in_pattern)

    batch_size = 600
    multiprocess = False

    def get_images_with_filenames(filenames):
        inputs, targets = load_images(filenames, deterministic=True)
        new_filenames = []
        for fname in filenames:
        	for i in range(int(len(inputs)/len(filenames))):
        		new_filenames.append(fname)
        #print 'inputs:',len(inputs),'filenames:',len(filenames),'new_filenames:',len(new_filenames)
        return inputs, targets, new_filenames


    gen = ParallelBatchIterator(get_images_with_filenames,
                                        filenames, ordered=True,
                                        batch_size=batch_size//3,
                                        multiprocess=multiprocess)

    predictions_file = os.path.join(model_folder, 'predictions_subset{}_epoch{}_model{}.csv'.format(subsets,epoch,P.MODEL_ID))
    candidates = pd.read_csv('../../data/candidates_V2.csv')
    candidates['probability'] = float(1337)

    print "Predicting {} patches".format(len(filenames))


    all_probabilities = []
    all_filenames = []

    n_batches = 0
    err_total = 0
    acc_total = 0

    for i, batch in enumerate(tqdm(gen)):
        inputs, targets, fnames = batch
        #print len(inputs)
        #print len(list(set(filenames)))
        targets = np.array(np.argmax(targets, axis=1), dtype=np.int32)
        err, l2_loss, acc, predictions, predictions_raw = val_fn(inputs, targets)
        err_total += err
        acc_total += acc
        all_probabilities += list(predictions_raw)
        all_filenames += list(fnames)

        n_batches += 1

    print "Loss", err_total / n_batches
    print "Accuracy", acc_total / n_batches

    submission = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','probability'])
    submission_row = 1;

    d = {f:[] for f in filenames}

    print "Grouping probabilities"
    for probability, f in zip(all_probabilities, all_filenames):
        d[f].append(probability)

    print "Filling predictions dataframe"
    for x in tqdm(d.iteritems()):
        fname, probabilities = x
        prob = np.mean(probabilities)
        candidates_row = int(os.path.split(fname)[1].replace('.pkl.gz','')) - 2
        candidates.set_value(candidates_row, 'probability', prob)
        submission.loc[candidates.index[submission_row]] = candidates.iloc[candidates_row]
        submission_row += 1

    #factor = len(all_filenames)/len(np.unique(all_filenames))
    #for fname in np.unique(all_filenames):
    #	prob = 0
    #	for i in range(len(all_filenames)):
    #		if all_filenames[i] == fname:
    #			prob += all_probabilities[i]
    #	prob /= factor
    #	candidates_row = int(os.path.split(fname)[1].replace('.pkl.gz','')) - 2
    #	candidates.set_value(candidates_row, 'probability', prob)
    #	submission.loc[candidates.index[submission_row]] = candidates.iloc[candidates_row]
    #	submission_row += 1

    #print submission
    submission_path = os.path.join(model_folder, 'predictions_subset{}_epoch{}_model{}.csv'.format(subsets,epoch,P.MODEL_ID))
    submission.to_csv(submission_path,columns=['seriesuid','coordX','coordY','coordZ','probability'],index=False)
