import unittest
from princ_comp_analysis import *


# TODO def transformToEigenRepresentation(mean_feature_vec, eigenvecs_of_training, data_to_eval):
# TODO def def hw1FindEigendigits(training_data):

class TestPrincipalComponentAnalysis(unittest.TestCase):

    def testHw1FindEigendigits(self):
        training_samples = np.array([[2, 4, 6], [3, 4, 8], [2, 14, 5], [11, 7, 6], [1, 0, 2]])
        expected_mean = np.array([4, 5, 7, 8, 1])

        training_samples_min_mean = np.array([[-2, 0, 2], [-2, -1, 3], [-5, 7, -2], [3, -1, -2], [0, -1, 1]])
        print(training_samples_min_mean @ np.transpose(training_samples_min_mean))
        expected_eig_vects = np.linalg.eigh(training_samples_min_mean @ np.transpose(training_samples_min_mean))
        print("full eig vecs")
        print(expected_eig_vects)
        print("Reduced ")
        print(np.linalg.eigh(np.transpose(training_samples_min_mean) @ training_samples_min_mean)[0])
        inv_eig_vects = training_samples_min_mean @ (np.linalg.eigh(np.transpose(training_samples_min_mean) @ training_samples_min_mean)[1])
        print(inv_eig_vects)

 #        expected_eig_vects = np.array([[ 0.10751498, -0.47188427, -0.42640143],
 # [ 0.00952176, -0.66603531, -0.63960215],
 # [ 0.95474001 , 0.1793466,   0.42640143],
 # [ 0.2592657,  -0.51367536,  0.42640143],
 # [ 0.09799322,  0.19415104, -0.21320072]])
 #
 #
 #

        result = hw1FindEigendigits(training_samples)
        self.assertTrue((expected_mean == result[0]).all())
        self.assertTrue(np.allclose(expected_eig_vects, result[1]))


    def testTransformToEigenRepresentation(self):
        pass


if __name__ == '__main__':
    unittest.main()
