from __future__ import division
import numpy as np
import os
import pickle
import glob
import Image
from skimage.io import imread
from sklearn.cross_validation import train_test_split

dataset_dir = "../../data/samples"

def load():

    tps = glob.glob(dataset_dir+"/*true.jpg")
    fps_2 = glob.glob(dataset_dir+"/*false.jpg")
    fps = np.random.choice(fps_2,10000)
    images_tps = [[imread(x)] for x in tps]
    images_fps = [[imread(x)] for x in fps]
    labels = np.concatenate((np.ones((len(images_tps))),np.zeros((len(images_fps))))).astype("ubyte")
    images = np.concatenate((images_tps,images_fps)).astype("float32")
    train_X, test_X, train_y, test_y = train_test_split(images,labels, test_size=0.4, random_state=1337)
    half = 0.5*len(test_X)
    val_X = test_X[:half]
    val_y = test_y[:half]
    test_X = test_X[half:]
    test_y = test_y[half:]
    label_to_names = {0:"false",1:"true"}

    # training set, batches 1-4
    # train_X = np.zeros((40000, 3, 32, 32), dtype="float32")
    # train_y = np.zeros((40000, 1), dtype="ubyte").flatten()
    # n_samples = 10000 # number of samples per batch
    # for i in range(0,4):
    #     f = open(os.path.join(dataset_dir, "data_batch_"+str(i+1)+""), "rb")
    #     cifar_batch = pickle.load(f)
    #     f.close()
    #     train_X[i*n_samples:(i+1)*n_samples] = (cifar_batch['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
    #     train_y[i*n_samples:(i+1)*n_samples] = np.array(cifar_batch['labels'], dtype='ubyte')
    #
    # # validation set, batch 5
    # f = open(os.path.join(dataset_dir, "data_batch_5"), "rb")
    # cifar_batch_5 = pickle.load(f)
    # f.close()
    # val_X = (cifar_batch_5['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
    # val_y = np.array(cifar_batch_5['labels'], dtype='ubyte')
    #
    # # labels
    # f = open(os.path.join(dataset_dir, "batches.meta"), "rb")
    # cifar_dict = pickle.load(f)
    # label_to_names = {k:v for k, v in zip(range(10), cifar_dict['label_names'])}
    # f.close()
    #
    # # test set
    # f = open(os.path.join(dataset_dir, "test_batch"), "rb")
    # cifar_test = pickle.load(f)
    # f.close()
    # test_X = (cifar_test['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
    # test_y = np.array(cifar_test['labels'], dtype='ubyte')
    #
    #
    # print("training set size: data = {}, labels = {}".format(train_X.shape, train_y.shape))
    # print("validation set size: data = {}, labels = {}".format(val_X.shape, val_y.shape))
    # print("test set size: data = {}, labels = {}".format(test_X.shape, test_y.shape))
    #
    return train_X, train_y, val_X, val_y, test_X, test_y, label_to_names