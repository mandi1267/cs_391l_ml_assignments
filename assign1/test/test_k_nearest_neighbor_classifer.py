import unittest
from k_nearest_neighbor_classifier import *

class TestKNearestNeighborClassifier(unittest.TestCase):

    def testGetDistBetweenSamples(self):

        classifier_inst = KNearestNeighborClassifier(5, 5)

        sample_pair_1a = np.array([1, 6, 9])
        sample_pair_1b = np.array([4, 6, 5])

        self.assertEqual(5, classifier_inst.getDistBetweenSamples(sample_pair_1a, sample_pair_1b))

    def testGetUniqueMode(self):

        classifier_inst = KNearestNeighborClassifier(5, 5)

        list_1 = np.array([[1], [2], [3], [4], [1]])
        list_with_many_modes = np.array([[2], [1], [3], [4]])
        list_with_two_modes = np.array([[3], [4], [2], [3], [4], [1]])

        self.assertEqual(1, classifier_inst.getUniqueMode(list_1))
        self.assertEqual(2, classifier_inst.getUniqueMode(list_with_many_modes))
        self.assertEqual(3, classifier_inst.getUniqueMode(list_with_two_modes))

    def testClassifySample(self):
        one_nearest_neighbor_classifier = KNearestNeighborClassifier(1, 3)

        training_data_1 = np.array([[1, 0, 100, 101], [2, 11, 100, 101], [3, -6, 100, 101]])
        training_labels_1 = np.array([1, 2, 3, 3])

        test_sample_class_1 = np.array([4, 14, 7])
        test_sample_class_2 = np.array([1, 10, -5])
        test_sample_class_3 = np.array([100, 100, 100])
        one_nearest_neighbor_classifier.trainClassifier(training_data_1, training_labels_1)

        self.assertEqual(3, one_nearest_neighbor_classifier.classifySample(test_sample_class_3))
        self.assertEqual(1, one_nearest_neighbor_classifier.classifySample(test_sample_class_1))
        self.assertEqual(2, one_nearest_neighbor_classifier.classifySample(test_sample_class_2))

if __name__ == '__main__':
    unittest.main()
