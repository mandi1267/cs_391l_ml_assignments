from functools import partial
import numpy as np
import time


class Classifier(object):

    def __init__(self, classifier_name):
        self.classifier_name = classifier_name

    def trainClassifier(self, training_data, training_labels):
        """
        Should train classifier and return info about how long it takes to do so.

        Must be implemented by subclass

        TODO: Still need to figure out what return should be
        """
        pass

    # TODO should this be in classifier or outside
    def testClassifier(self, test_data, test_labels):

        classifier_start = time.time()

        # Compute the output label for each sample based on the particular classifier type
        output_labels = np.apply_along_axis(self.classifySample, 0, test_data)

        # Compute the difference between the actual labels and the labels
        # output by the classifier
        # Entries that are 0 will indicate correctly classified test samples
        # whereas non-zero entries will indicate a difference betweent the
        # true label and the generated label
        correct_label_vec = test_labels - output_labels

        # Count the number of incorrectly classified (non-zero) entries
        incorrect_classification_count = np.count_nonzero(correct_label_vec)

        # The error rate is the number of incorrectly classified entries over
        # the total number
        error_rate = float(incorrect_classification_count) / np.shape(test_labels)[0]

        # TODO what do we want to return? Just error rate? Time to evaluate?
        end_time = time.time()

        time_per_sample = (end_time - classifier_start) / test_labels.size
        return (error_rate, time_per_sample)


    def classifySample(self, sample):
        pass

    def getName(self):
        return self.classifier_name
