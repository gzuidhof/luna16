from __future__ import division
import sys
import params
import numpy as np
import os
import skimage.io
model_folder = '../../models/'



#Run with: python predict_resnet.py 1466485849_resnet 194 9
# To predict subset 9 of the 1466485849_resnet model at epoch 194
# To predict multiple subsets just add them in succession (such as 123 or 09)

if len(sys.argv) < 3:
    print "Missing arguments, first argument is model name, second is epoch, third is which subsets to predict"
    quit()

model_folder = os.path.join(model_folder, sys.argv[1])

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
    import augment

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

    print "Defining updates.."
    train_fn, val_fn, l_r = resnet.define_updates(network, input_var, target_var)

    #in_pattern = '../../data/cadV2_0.5mm_64x64_xy_xz_yz/subset[{}]/*/*.pkl.gz'.format(subsets)
    in_pattern = '/scratch-shared/vdgugten/data/cadOWN_0.5mm_96x96_xy_xz_yz/subset[{}]/*/*.pkl.gz'.format(subsets)
    filenames = glob(in_pattern)

    batch_size = 1200
    multiprocess = True

    test_im = np.zeros((64,64))
    n_testtime_augmentation = len(augment.testtime_augmentation(test_im, 0)[0])

    def get_images_with_filenames(filenames):
        inputs, targets = load_images(filenames, deterministic=True)

        new_inputs = []
        new_targets = []

        for image, target in zip(inputs, targets):
            ims, trs = augment.testtime_augmentation(image[0], target) #Take color channel of image
            new_inputs += ims
            new_targets+=trs

        new_filenames = []
        for fname in filenames:
            for i in range(int(len(new_inputs)/len(filenames))):
                new_filenames.append(fname)
        #print 'inputs:',len(inputs),'filenames:',len(filenames),'new_filenames:',len(new_filenames)
        return np.array(new_inputs,dtype=np.float32),np.array(new_targets,dtype=np.int32), new_filenames


    gen = ParallelBatchIterator(get_images_with_filenames,
                                        filenames, ordered=True,
                                        batch_size=batch_size//(3*n_testtime_augmentation),
                                        multiprocess=multiprocess, n_producers=11)

    predictions_file = os.path.join(model_folder, 'predictions_subset{}_epoch{}_model{}.csv'.format(subsets,epoch,P.MODEL_ID))
    candidates = pd.read_csv('../../csv/unetRelabeled.csv')
    #candidates['probability'] = float(1337)

    print "Predicting {} patches".format(len(filenames))


    all_probabilities = []
    all_filenames = []

    n_batches = 0
    err_total = 0
    acc_total = 0

    for i, batch in enumerate(tqdm(gen)):
        inputs, targets, fnames = batch
        targets = np.array(np.argmax(targets, axis=1), dtype=np.int32)
        err, l2_loss, acc, predictions, predictions_raw = val_fn(inputs, targets)
        err_total += err
        acc_total += acc
        all_probabilities += list(predictions_raw)
        all_filenames += list(fnames)

        n_batches += 1

    print "Loss", err_total / n_batches
    print "Accuracy", acc_total / n_batches

    d = {f:[] for f in filenames}

    print "Grouping probabilities"
    for probability, f in zip(all_probabilities, all_filenames):
        d[f].append(probability)

    print "Filling predictions dataframe"

    data = []
    # for x in tqdm(d.iteritems()):
    #     fname, probabilities = x
    #     prob = np.mean(probabilities)
    #     candidates_row = int(os.path.split(fname)[1].replace('.pkl.gz','')) - 2
    #     candidates.set_value(candidates_row, 'probability', prob)
    #     submission.loc[candidates.index[submission_row]] = candidates.iloc[candidates_row]
    #     submission_row += 1
    for x in tqdm(d.iteritems()):
        fname, probabilities = x
        prob = np.mean(probabilities)
        candidates_row = int(os.path.split(fname)[1].replace('.pkl.gz','')) - 2
        #print candidates.iloc[candidates_row].values
        data.append(list(candidates.iloc[candidates_row].values)[:-1]+[str(prob)])

    submission = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','probability'],data=data)


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
    # Z HAVE BEEN SWAPPED TO MATCH SUBMISSION
    submission.to_csv(submission_path,columns=['seriesuid','coordX','coordY','coordZ','probability'])
