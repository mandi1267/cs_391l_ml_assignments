
from k_nearest_neighbor_classifier import *
import matplotlib.pyplot as plt

import numpy as np

def readMNISTData(data_file_name):
    # TODO implement something to read the dataset into a numpy array
    pass


def readMNISTLabels(labels_file_name):
    # TODO implement something to read the labels for the dataset into a numpy vector
    pass

def extractEigenVectors(training_data):
    """
    Compute the vector by which to multiply data to express it in M values (
    which correspond to the directions along the M largest eigenvectors) of the
    training data.

    Args:
        training_data (2D numpy array): Array containing training data. Each
            sample corresponds to one row.

    Returns:
        NxM matrix giving the M largest eigenvectors of X^T * X (where X is the
        training data, N is the number of features, and M is the number of
        samples)
    """

    training_transpose = np.transpose(training_data)

    # Get the MxM matrix (training data has m rows, so the transpose will have n rows)
    m_by_m_mat = training_data.dot(training_transpose)

    # Get the eigen vectors of the MxM matrix
    # Because a matrix times its transpose is symmetric, we can use eigh
    eig_vals, eig_vecs = np.linalg.eigh(m_by_m_mat)

    m_largest_eig_vects_of_n_by_n_mat = training_transpose.dot(eig_vecs)
    return m_largest_eig_vects_of_n_by_n_mat

def transformToEigenRepresentation(eigenvector_transform_info, data_to_eval):

    # TODO Do we want this to transform a single piece of data or several rows
    # of data (samples)

    """
    Transform the input data to the eigenvector representation (so it has M
    features instead of N, where N is the original feature count and M is the
    number of training samples).

    Args:
        eigenvector_transform_info (2D numpy array): Matrix with each column
            being an eigenvector of the X^T * X (where X is the training data).
            Using this to transform the data to evaluate to have M features
            instead of N.
        data_to_eval (2D numpy array): Has N columns and Z rows (Z is the
            number of samples to transform).


    Returns:
        ZxM matrix with each sample now having M features instead of N (based
        on the projection onto the eigenvectors of the training data).
    """

    return data_to_eval.dot(eigenvector_transform_info)

# Using the special variable
# __name__
if __name__=="__main__":
    training_data_file_name = "" # TODO
    training_label_file_name = "" # TODO

    test_data_file_name = "" # TODO
    test_label_file_name = "" # TODO

    training_data = readMNISTData(training_data_file_name)
    training_labels = readMNISTLabels(training_label_file_name)

    test_data = readMNISTData(test_data_file_name)
    test_labels = readMNISTLabels(test_label_file_name)

    # TODO remove
    training_data = np.arange(40).reshape(4, 10)
    training_labels = np.arange(4)
    test_data = (np.arange(20) + 0.1).reshape(2, 10)
    test_labels = np.arange(2)

    # Find the eigenvector representation from the training data
    eigenvector_transform_info = extractEigenVectors(training_data)
    transformed_training_data = transformToEigenRepresentation(eigenvector_transform_info, training_data)

    # Also transform the test data
    transformed_test_data = transformToEigenRepresentation(eigenvector_transform_info, test_data)

    # Construct classifiers to evaluate
    k_nearest_neighbor_classifiers = {i:KNearestNeighborClassifier(i) for i in range(1, (np.shape(training_labels)[0]) + 1)}
    classifiers = k_nearest_neighbor_classifiers.values() # TODO populate once have classifiers to evaluate (consider trying out other classifiers too?)

    # TODO try to get python3 working. Issue currently is can't find scipy
    classifiers_dict = {classifier.getName():classifier for classifier in classifiers}
    classifier_train_info = {}
    for classifier_name, classifier in classifiers_dict.items():
        classifier_train_info[classifier_name] = classifier.trainClassifier(transformed_training_data, training_labels)

    classifier_test_info = {}
    for classifier_name, classifier in classifiers_dict.items():
        classifier_test_info[classifier_name] = classifier.testClassifier(transformed_test_data, test_labels)

    print(classifier_test_info)

    best_knn_k_num = 0
    best_knn_error_rate = 1
    knn_results_x = k_nearest_neighbor_classifiers.keys()
    knn_results_y = []
    for k_num in k_nearest_neighbor_classifiers.keys():
        kNN_classifier_name = k_nearest_neighbor_classifiers[k_num].getName()
        classifier_results = classifier_test_info[kNN_classifier_name]
        if (best_knn_error_rate > classifier_results):
            best_knn_k_num = k_num
            best_knn_error_rate = classifier_results
        knn_results_y.append(classifier_results)

    plt.plot(knn_results_x, knn_results_y, 'rD-.')

    plt.xlabel("K in KNN")
    plt.ylabel("Error Rate")
    plt.title("Error Rate by K Value")
    plt.legend()

    plt.show()



    # TODO plot results
