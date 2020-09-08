import unittest
from classifier_interface import *

class TestClassifierImpl(Classifier):
    def __init__(self, list_of_labels_to_return):
        super(TestClassifierImpl, self).__init__("Dummy classifier for test")
        self.list_of_labels_to_return = list_of_labels_to_return
        self.next_label_index = 0
        self.samples_given = []

    def trainClassifier(self, training_data, training_labels):
        pass

    def classifySample(self, sample):
        return_label = self.list_of_labels_to_return[self.next_label_index]
        self.next_label_index += 1
        if (self.next_label_index == len(self.list_of_labels_to_return)):
            self.next_label_index = 0
        self.samples_given.append(sample)
        return return_label

    def getSamplesGiven(self):
        return_data = self.samples_given
        self.samples_given = []
        return return_data


class TestClassifier(unittest.TestCase):

    def testTestClassifier(self):
        list_of_labels_to_return = [1, 2, 1, 2, 3, 5, 9, 1, 4, 3]

        true_labels_full_accuracy = np.array(list_of_labels_to_return)
        true_labels_no_accuracy = np.array([i - 1 for i in list_of_labels_to_return])
        two_modified_list_of_labels = list_of_labels_to_return[:]
        two_modified_list_of_labels[5] += 1
        two_modified_list_of_labels[7] += 3
        true_labels_eighty_perc_accurate = np.array(two_modified_list_of_labels)

        test_classifier = TestClassifierImpl(list_of_labels_to_return)

        samples = np.arange(30).reshape(3, 10)
        self.assertEqual(1, test_classifier.testClassifier(samples, true_labels_full_accuracy)[0])
        samples_given_to_classifier = test_classifier.getSamplesGiven()
        for i in range(samples.shape[1]):
            self.assertTrue((samples_given_to_classifier[i] == samples[:, i]).all())

        self.assertEqual(0, test_classifier.testClassifier(samples, true_labels_no_accuracy)[0])
        samples_given_to_classifier = test_classifier.getSamplesGiven()
        for i in range(samples.shape[1]):
            self.assertTrue((samples_given_to_classifier[i] == samples[:, i]).all())

        self.assertEqual(0.8, test_classifier.testClassifier(samples, true_labels_eighty_perc_accurate)[0])
        samples_given_to_classifier = test_classifier.getSamplesGiven()
        for i in range(samples.shape[1]):
            self.assertTrue((samples_given_to_classifier[i] == samples[:, i]).all())

if __name__ == '__main__':
    unittest.main()
