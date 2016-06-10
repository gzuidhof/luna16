from __future__ import division
import numpy as np

def _calc_errors(truth, prediction, class_number=1):
    tp = np.sum(np.equal(truth,class_number)*np.equal(prediction,class_number))
    tn = np.sum(np.not_equal(truth,class_number)*np.not_equal(prediction,class_number))

    fp = np.sum(np.not_equal(truth,class_number)*np.equal(prediction,class_number))
    fn = np.sum(np.equal(truth,class_number)*np.not_equal(prediction,class_number))

    return tp, tn, fp, fn

class Metrics(object):
    """ Keeps track of metrics per epoch
    """

    def __init__(self, name, metric_names, n_classes):
        self.name = name
        self.metric_names = metric_names
        self.n_classes = n_classes

        # One value per epoch
        self.values = []

        # One value per batch
        self.batch_values = []
        self.batch_errors = {n:[] for n in range(n_classes)}

        self.labels = []

    def append(self, metrics):
        self.batch_values.append(metrics)

    def append_prediction(self, truth, prediction):
        for c in range(self.n_classes):
            self.batch_errors[c].append(_calc_errors(truth, prediction, c))

    def batch_done(self, skip_classes=[0]):
        n_batches = len(self.batch_values)
        values = list(np.sum(np.array(self.batch_values),axis=0)/n_batches)
        labels = self.metric_names

        classes = range(self.n_classes)
        classes = filter(lambda x: x not in skip_classes, classes)

        # Prediction and truth have been appended at least once,
        # so we can add precision/recall
        if len(self.batch_errors[0]) > 0:
            for c in classes:
                errors = np.array(self.batch_errors[c])
                tp, tn, fp, fn = np.sum(errors, axis=0)

                specificity = tn/(fp+tn)
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)

                values += [specificity, precision, recall]
                new_labels = ["Specificity", "Precision", "Recall"]

                if len(classes) > 1:
                    new_labels = [l+"_class"+str(c) for l in new_labels]

                labels += new_labels

        self.values.append(values)
        self.batch_values = []
        self.batch_errors = {n:[] for n in range(self.n_classes)}
        self.labels = labels

        return labels, values

    def values_per_epoch(self):
        return self.labels, zip(*self.values)
