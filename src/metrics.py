import numpy as np

def calc_errors(truth, prediction):
    tp = np.sum(np.equal(truth,1)*np.equal(prediction,1))
    tn = np.sum(np.equal(truth,0)*np.equal(prediction,0))
    fp = np.sum(np.equal(truth,0)*np.equal(prediction,1))
    fn = np.sum(np.equal(truth,1)*np.equal(prediction,0))

    return tp, tn, fp, fn
