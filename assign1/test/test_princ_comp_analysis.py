import unittest
from princ_comp_analysis import *


class TestPrincipalComponentAnalysis(unittest.TestCase):

    def testHw1FindEigendigits(self):
        training_samples = np.array([[2, 4, 6], [3, 4, 8], [2, 14, 5], [11, 7, 6], [1, 0, 2]])
        expected_mean = np.array([4, 5, 7, 8, 1])

        training_samples_min_mean = np.array([[-2, 0, 2], [-2, -1, 3], [-5, 7, -2], [3, -1, -2], [0, -1, 1]])
        expected_eig_vects = np.linalg.eigh(training_samples_min_mean @ np.transpose(training_samples_min_mean))
        inv_eig_vects = training_samples_min_mean @ (np.linalg.eigh(np.transpose(training_samples_min_mean) @ training_samples_min_mean)[1])

        result = hw1FindEigendigits(training_samples)
        self.assertTrue((expected_mean == result[0]).all())

if __name__ == '__main__':
    unittest.main()
