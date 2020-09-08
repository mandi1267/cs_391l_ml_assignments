from functools import partial
import numpy as np
import time

class Classifier(object):
    """
    Interface defining what methods are needed to evaluate a classification algorithm and for a classifier to
    classify samples
    """

    def __init__(self, classifier_name):
        """
        Constructor for a generic classifier.

        Args:
            classifier_name (string): Name of the classifier for display and identification purposes
        """
        self.classifier_name = classifier_name

    def trainClassifier(self, training_data, training_labels):
        """
        Should train classifier. Must be implemented by subclass.

        Args:
            training_data (2D numpy array):     2D array of size P x K containing the training data, where each sample
                                                is a column.
            training_labels (1D numpy array):   1D array of length K containing the labels for the training data.
        """
        pass

    def testClassifier(self, test_data, test_labels):
        """
        Test the classifier based on its performance on the given test data and test labels.

        Args:
            test_data (2D numpy array):     2D array of size P x K containing the samples on which to test the
                                            classifier.
            test_labels (1D numpy array):   Ground truth labels for the test data.
        Returns:
            Tuple containing the accuracy rate and the average time needed to classify each sample. Accuracy rate is 1 if the
            classifier is always correct and 0 if the classifier is always wrong.
        """

        classifier_start = time.time()

        # Compute the output label for each sample
        output_labels = np.apply_along_axis(self.classifySample, 0, test_data)

        classifier_end_time = time.time()

        # Compute the difference between the actual labels and the labels
        # output by the classifier
        # Entries that are 0 will indicate correctly classified test samples
        # whereas non-zero entries will indicate a difference betweent the
        # true label and the generated label
        correct_label_vec = test_labels - output_labels

        # Count the number of incorrectly classified (non-zero) entries
        incorrect_classification_count = np.count_nonzero(correct_label_vec)

        # The accuracy rate is the number of correctly classified entries over
        # the total number
        accuracy_rate = (np.shape(test_labels)[0] - float(incorrect_classification_count)) / np.shape(test_labels)[0]

        # Compute the time to classify each sample
        time_per_sample = (classifier_end_time - classifier_start) / test_labels.size
        return (accuracy_rate, time_per_sample)


    def classifySample(self, sample):
        """
        Classify the given sample. Must be implemented by the subclass.

        Args:
            sample (1D numpy array): Sample to classify.
        Returns:
            Label for the sample.
        """
        pass
