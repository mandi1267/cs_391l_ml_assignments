
from k_nearest_neighbor_classifier import *
from mnist_reader import *
from display_utils import *
import matplotlib.pyplot as plt
import csv

import numpy as np
from math import ceil
from math import floor
import sys
import datetime

class KnnResults:
    def __init__(self, num_training_samples, num_principal_components, num_nearest_neighbors, error_rate, classification_time_per_sample):
        self.num_training_samples = num_training_samples
        self.num_principal_components = num_principal_components
        self.num_nearest_neighbors = num_nearest_neighbors
        self.error_rate = error_rate
        self.classification_time_per_sample = classification_time_per_sample

def hw1FindEigendigits(training_data):
    """
    Compute the vector by which to multiply data to express it in M values (
    which correspond to the directions along the M largest eigenvectors) of the
    training data.

    Assumes that there are less samples than there are featurs.

    Args:
        training_data (2D numpy array): Array containing training data. Each
            sample corresponds to one column.

    Returns:
        Tuple containing vector of length N (number of pixels) containing mean
        value of training samples for each pixel and NxM matrix containing the
        M highest eighenvectors (sorted in descending order by their
        eigenvalue), where M is the number of training samples. Each column is
        an eigenvector.
    """
    mean_vec = np.mean(training_data, axis=1)

    mean_0_training_data = training_data - mean_vec[:, None]
    mean_0_training_data_transpose = np.transpose(mean_0_training_data)

    # Get the MxM matrix (training data has N rows, so the transpose will have M rows)
    m_by_m_mat = mean_0_training_data_transpose.dot(mean_0_training_data)

    # Get the eigen vectors of the MxM matrix
    # Because a matrix times its transpose is symmetric, we can use eigh
    eig_vals, eig_vecs = np.linalg.eigh(m_by_m_mat)

    # Each column in eig_vecs is an eigenvector

    # Sort the eigenvectors by their corresponding eigenvalues in descending
    # order
    # Transpose the eigenvector matrix (so each eigenvector is a row) and
    # add a column at the end corresponding to the eigenvalue
    eig_vals_and_vecs_matrix = np.column_stack((np.transpose(eig_vecs), eig_vals))

    # Sort the combined matrix by the last column and then remove the
    # eigenvalue column
    sorted_eig_vecs_and_vals = eig_vals_and_vecs_matrix[(-1 * eig_vals_and_vecs_matrix[:, -1]).argsort()]
    sorted_eig_vecs = sorted_eig_vecs_and_vals[:, :-1]

    # Transform the eigenvectors so that they are eigenvectors of the training
    # data
    sorted_eig_vecs_in_columns = np.transpose(sorted_eig_vecs)
    m_largest_eig_vecs_of_n_by_n_mat = mean_0_training_data.dot(sorted_eig_vecs_in_columns)
    eig_vector_magnitudes = np.linalg.norm(m_largest_eig_vecs_of_n_by_n_mat, axis=0)
    normalized_m_largest_eig_vecs_of_n_by_n_mat = m_largest_eig_vecs_of_n_by_n_mat / (eig_vector_magnitudes[None, :])

    return mean_vec, normalized_m_largest_eig_vecs_of_n_by_n_mat

def transformToEigenRepresentation(mean_feature_vec, eigenvals_of_training, data_to_eval):

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
        data_to_eval (2D numpy array): Has Z columns and N rows (Z is the
            number of samples to transform).


    Returns:
        ZxM matrix with each sample now having M features instead of N (based
        on the projection onto the eigenvectors of the training data).
    """

    mean_subtracted_data = data_to_eval - mean_feature_vec[:, None]
    return np.transpose(np.transpose(mean_subtracted_data).dot(eigenvals_of_training))



def getRandomSubsetOfData(data, labels, number_of_samples_to_return):
    indices = np.random.choice(labels.size, number_of_samples_to_return, replace=False)
    label_subset = labels[indices]
    data_subset = data[:, indices]
    return (data_subset, label_subset)

def plotResults(k_nn_results):
    # Get all unique training sample values
    training_sample_counts = sorted(list(set([knn_result.num_training_samples for knn_result in k_nn_results])))

    # Training values mapped to the best error rate to the training value and the results object corresponding to the
    # error rate
    best_results_by_training_sample_count = {training_value:(1, None) for training_value in training_sample_counts}

    # Get all unique feature count values
    feature_count_values = sorted(list(set([knn_result.num_principal_components for knn_result in k_nn_results])))

    # Feature counts mapped to the best error rate to the training value and the results object corresponding to the
    # error rate
    best_results_by_feature_count = {feature_count:(1, None) for feature_count in feature_count_values}

    # Get all unique k values
    nearest_neighbor_values = sorted(list(set([knn_result.num_nearest_neighbors for knn_result in k_nn_results])))

    # Nearest neighbor number mapped to the best error rate to the training value and the results object corresponding
    # to the error rate
    best_results_by_neighbor_count = {neighbor_count:(1, None) for neighbor_count in nearest_neighbor_values}

    for k_nn_result in k_nn_results:
        if (best_results_by_training_sample_count[k_nn_result.num_training_samples][0] > k_nn_result.error_rate):
            best_results_by_training_sample_count[k_nn_result.num_training_samples] = (k_nn_result.error_rate, k_nn_result)
        if (best_results_by_feature_count[k_nn_result.num_principal_components][0] > k_nn_result.error_rate):
            best_results_by_feature_count[k_nn_result.num_principal_components] = (k_nn_result.error_rate, k_nn_result)
        if (best_results_by_neighbor_count[k_nn_result.num_nearest_neighbors][0] > k_nn_result.error_rate):
            best_results_by_neighbor_count[k_nn_result.num_nearest_neighbors] = (k_nn_result.error_rate, k_nn_result)

    # Plot the best results for each training sample count
    error_rate_by_training_sample_count = [best_results_by_training_sample_count[training_count][0] for training_count in training_sample_counts]
    error_rate_by_feature_count = [best_results_by_feature_count[feature_count_value][0] for feature_count_value in feature_count_values]
    error_rate_by_num_nearest_neighbor = [best_results_by_neighbor_count[neighbor_count][0] for neighbor_count in nearest_neighbor_values]

    plt.plot(training_sample_counts, error_rate_by_training_sample_count)
    plt.xlabel("Number of training samples")
    plt.ylabel("Error rate")
    plt.title("Best Error Rates Across All Counts of Principal Component and Neighbor Values by Number of Training Examples")
    plt.show()

    plt.plot(feature_count_values, error_rate_by_feature_count)
    plt.xlabel("Number of principal components")
    plt.ylabel("Error rate")
    plt.title("Best Error Rates Across All Training Set Sizes and Neighbor Values by Number of Principal Components")
    plt.show()

    plt.plot(nearest_neighbor_values, error_rate_by_num_nearest_neighbor)
    plt.xlabel("Number of nearest neighbors")
    plt.ylabel("Error rate")
    plt.title("Best Error Rates Across All Training Set Sizes and Counts of Principal Component by Number of Nearest Neighbors")
    plt.show()

def writeResultsToFile(k_nn_results, output_file_name):
    with open(output_file_name, 'w') as results_file:
        header_line = ["num_training_samples", "num_principal_components", "num_nearest_neighbors", "error_rate", "classification_time_per_sample"]
        csv_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header_line)
        for result in k_nn_results:
            csv_writer.writerow([result.num_training_samples, result.num_principal_components, result.num_nearest_neighbors, result.error_rate, result.classification_time_per_sample])

if __name__=="__main__":

    if (len(sys.argv) >= 5):
        print("Getting training and test data from the command line")
        training_data_file_name = sys.argv[1]
        training_label_file_name = sys.argv[2]

        test_data_file_name = sys.argv[3]
        test_label_file_name = sys.argv[4]


    else:
        training_data_file_name = "/Users/mandiadkins/Downloads/train-images.idx3-ubyte"
        training_label_file_name = "/Users/mandiadkins/Downloads/train-labels.idx1-ubyte"

        test_data_file_name = "/Users/mandiadkins/Downloads/t10k-images.idx3-ubyte"
        test_label_file_name = "/Users/mandiadkins/Downloads/t10k-labels.idx1-ubyte"

    training_data, num_rows_in_img, num_cols_in_img = readMNISTData(training_data_file_name)
    training_labels = readMNISTLabels(training_label_file_name)

    for i in range(10):
        data = training_data[i, :]
        image_version = convertVectorToImage(data, num_rows_in_img, num_cols_in_img)
        # displayImage(image_version, training_labels[i])

    # Load the test data
    test_data, num_rows_in_img, num_cols_in_img = readMNISTData(test_data_file_name)
    test_labels = readMNISTLabels(test_label_file_name)

    img_vector_size = num_rows_in_img * num_cols_in_img

    for i in range(10):
        data = test_data[i, :]
        image_version = convertVectorToImage(data, num_rows_in_img, num_cols_in_img)
        # displayImage(image_version, test_labels[i])

    # Transpose the training and test data so that each image is a column
    training_data = np.transpose(training_data)
    test_data = np.transpose(test_data)

    num_training_size_increment = 50
    num_samples_to_use = [min(num_training_size_increment * i, img_vector_size)  for i in range(1, 1 + ceil(img_vector_size / num_training_size_increment))]
    # num_samples_to_use = [784]
    # num_samples_to_use = [105]
    print(num_samples_to_use)

    test_data_subset, test_label_subset = getRandomSubsetOfData(training_data, training_labels, 1000)

    error_rate_by_train_set_size_and_feature_count = {}

    # Make map where outer key is training set size,

    k_nn_results = []
    results_by_feature_count = {}

    for train_sample_size in num_samples_to_use:
        train_data_subset, train_label_subset = getRandomSubsetOfData(training_data, training_labels, train_sample_size)
        print(train_data_subset.shape)

        print("Training sample size " + str(train_sample_size))

        feature_count_inc_size = 25

        max_single_inc_range = 30
        max_small_inc_range = 75
        small_inc = 5

        num_features_to_try = [i + 1 for i in range(max_single_inc_range) if i < train_sample_size]
        num_features_to_try.extend([i for i in range(max_single_inc_range + small_inc, max_small_inc_range, small_inc) if i < train_sample_size])
        if (train_sample_size >= max_small_inc_range):
            num_features_to_try.extend([i for i in range(max_small_inc_range, train_sample_size + 1, feature_count_inc_size)])
            if (train_sample_size not in num_features_to_try):
                num_features_to_try.append(train_sample_size)
        # num_features_to_try = [10, 20]

        # Find the eigenvector representation from the training data
        mean_feature_vec, eigenvectors_of_train = hw1FindEigendigits(train_data_subset)

        print("Features to try")
        print(num_features_to_try)

        for feature_count in num_features_to_try:
            print("Train sample size: " + str(train_sample_size) + ", Feature count " + str(feature_count))
            reduced_eigenvectors_of_train = eigenvectors_of_train[:, :feature_count]
            transformed_training_data = transformToEigenRepresentation(mean_feature_vec, reduced_eigenvectors_of_train, train_data_subset)

            # Also transform the test data
            transformed_test_data = transformToEigenRepresentation(mean_feature_vec, reduced_eigenvectors_of_train, test_data_subset)

            # Construct classifiers to evaluate
            k_val_inc = 5
            max_single_inc_range = 15
            k_vals_to_try = [i + 1 for i in range(max_single_inc_range)]
            max_k_val = min(train_sample_size, 100)
            k_vals_to_try.extend([min(max_k_val, k_val_inc * i) for i in range((1 + floor(max_single_inc_range / k_val_inc)), 1 + ceil(max_k_val / k_val_inc))])
            print("K Values to Evaluate")
            print(k_vals_to_try)
            # k_vals_to_try = [min(k_val_inc * i, feature_count) for i in range(max_sin, 1 + ceil(feature_count / k_val_inc))]
            k_nearest_neighbor_classifiers = {i:KNearestNeighborClassifier(i, feature_count) for i in k_vals_to_try}
            classifiers = k_nearest_neighbor_classifiers.values() # TODO populate once have classifiers to evaluate (consider trying out other classifiers too?)

            classifiers_dict = {classifier.getName():classifier for classifier in classifiers}
            classifier_train_info = {}
            for classifier_name, classifier in classifiers_dict.items():
                classifier_train_info[classifier_name] = classifier.trainClassifier(transformed_training_data, train_label_subset)

            classifier_test_info = {}
            for classifier_name, classifier in classifiers_dict.items():
                classifier_test_info[classifier_name] = classifier.testClassifier(transformed_test_data, test_label_subset)

            best_knn_k_num = 0
            best_knn_error_rate = 1
            knn_results_x = k_nearest_neighbor_classifiers.keys()
            knn_results_y = []

            for k_num in k_nearest_neighbor_classifiers.keys():

                kNN_classifier_name = k_nearest_neighbor_classifiers[k_num].getName()
                classifier_results = classifier_test_info[kNN_classifier_name]

                k_nn_results.append(KnnResults(train_sample_size, feature_count, k_num, classifier_results[0], classifier_results[1]))
                knn_results_y.append(classifier_results[0])

            results_by_feature_count[feature_count] = (knn_results_x, knn_results_y)

    output_file_name = "assign_1_results_" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".csv"
    writeResultsToFile(k_nn_results, output_file_name)

    for feature_count in num_features_to_try:
        results_for_feature_count = results_by_feature_count[feature_count]
        plt.plot(results_for_feature_count[0], results_for_feature_count[1], 'rD-.')

        plt.xlabel("K in KNN")
        plt.ylabel("Error Rate")
        plt.title("Error Rate by K Value for training set size " + str(train_sample_size) + " and " + str(feature_count) + " principal components")
        plt.legend()
        plt.show()

    plotResults(k_nn_results)
