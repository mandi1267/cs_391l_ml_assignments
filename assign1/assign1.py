"""
Main program for executing the demo and exhaustive versions of assignment 1 for CS 393R, Fall 2020.

Author: Amanda Adkins
"""

from k_nearest_neighbor_classifier import *
from mnist_reader import *
from display_utils import *
from princ_comp_analysis import *
from neural_net_classifier import *
import matplotlib.pyplot as plt
import csv

import numpy as np
from math import ceil
from math import floor
import sys
import datetime
import argparse

class KnnResults:
    def __init__(self, num_training_samples, num_principal_components, num_nearest_neighbors, accuracy_rate, classification_time_per_sample):
        """
        Constructor for the results.
        Captures the parameters for a particular classifier evaluation and the results of the evaluation.

        Args:
            num_training_samples (int):                 Number of training samples that were used to train the
                                                        classifier.
            num_principal_components (int):             Number of features that were used in representing the digits.
                                                        Equivalent to the number eigenvectors that were extracted from
                                                        the training data.
            num_nearest_neighbors (int):                K in the K-NN classifier. Each sample is classified according
                                                        to the mode of the K training samples that are closest to the
                                                        input sample.
            accuracy_rate (double):                     Accuracy rate of the classifier. 0 indicates that there were no
                                                        correct classifications, 1 indicates that every sample was
                                                        correctly classified.
            classification_time_per_sample (double):    Average amount of time it takes to classify each sample.
        """

        self.num_training_samples = num_training_samples
        self.num_principal_components = num_principal_components
        self.num_nearest_neighbors = num_nearest_neighbors
        self.accuracy_rate = accuracy_rate
        self.classification_time_per_sample = classification_time_per_sample

def getRandomSubsetOfData(data, labels, number_of_samples_to_return):
    """
    Get a random subset of the input data and corresponding labels.

    Args:
        data (2D numpy array):              Data to get the subset of. This is a P X L array, where P is the number of
                                            features in the data and L is the total number of samples. Each column is
                                            one sample.
        labels (1D numpy array):            Labels for the input data. Has L entries, where the first entry corresponds
                                            to the first column (sample) in the data, and so on.
        number_of_samples_to_return (int):  Number of total samples to return. Assumed to be less than or equal to the
                                            number of samples in the input data.
    Returns:
        A tuple of the following:
            P x number_of_samples_to_return array with a random selection of the input data, drawn without replacement.
                Each column is a sample
            number_of_samples_to_return length vector with the labels corresponding to the selected samples returned.
    """
    # Get the indices of the samples to return as a random sample
    indices = np.random.choice(labels.size, number_of_samples_to_return, replace=False)

    # Get the labels and data that corresponds to the randomly selected indices
    label_subset = labels[indices]
    data_subset = data[:, indices]
    return (data_subset, label_subset)

def plotResults(k_nn_results):
    """
    Plot the results for the exhaustive execution of assignment 1.

    Plots the best accuracy for each training set size, k value, and feature count.

    Args:
        k_nn_results (list of KnnResults):  List of results for every execution (each has a different permutation of
                                            training set size, feature count, and k-value).
    """
    # Get all unique training sample values
    training_sample_counts = sorted(list(set([knn_result.num_training_samples for knn_result in k_nn_results])))

    # Training values mapped to the best accuracy_rate to the training value and the results object corresponding to the
    # accuracy_rate
    best_results_by_training_sample_count = {training_value:(0, None) for training_value in training_sample_counts}

    # Get all unique feature count values
    feature_count_values = sorted(list(set([knn_result.num_principal_components for knn_result in k_nn_results])))

    # Feature counts mapped to the best accuracy_rate to the training value and the results object corresponding to the
    # accuracy_rate
    best_results_by_feature_count = {feature_count:(0, None) for feature_count in feature_count_values}

    # Get all unique k values
    nearest_neighbor_values = sorted(list(set([knn_result.num_nearest_neighbors for knn_result in k_nn_results])))

    # Nearest neighbor number mapped to the best accuracy_rate to the training value and the results object corresponding
    # to the accuracy_rate
    best_results_by_neighbor_count = {neighbor_count:(0, None) for neighbor_count in nearest_neighbor_values}

    for k_nn_result in k_nn_results:
        if (best_results_by_training_sample_count[k_nn_result.num_training_samples][0] < k_nn_result.accuracy_rate):
            best_results_by_training_sample_count[k_nn_result.num_training_samples] = (k_nn_result.accuracy_rate, k_nn_result)
        if (best_results_by_feature_count[k_nn_result.num_principal_components][0] < k_nn_result.accuracy_rate):
            best_results_by_feature_count[k_nn_result.num_principal_components] = (k_nn_result.accuracy_rate, k_nn_result)
        if (best_results_by_neighbor_count[k_nn_result.num_nearest_neighbors][0] < k_nn_result.accuracy_rate):
            best_results_by_neighbor_count[k_nn_result.num_nearest_neighbors] = (k_nn_result.accuracy_rate, k_nn_result)

    # Plot the best results for each training sample count
    accuracy_rate_by_training_sample_count = [best_results_by_training_sample_count[training_count][0] for training_count in training_sample_counts]
    accuracy_rate_by_feature_count = [best_results_by_feature_count[feature_count_value][0] for feature_count_value in feature_count_values]
    accuracy_rate_by_num_nearest_neighbor = [best_results_by_neighbor_count[neighbor_count][0] for neighbor_count in nearest_neighbor_values]

    plt.plot(training_sample_counts, accuracy_rate_by_training_sample_count)
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy rate")
    plt.title("Best Accuracy Rates Across All Counts of Principal Component and Neighbor Values by Number of Training Examples")
    plt.show()

    plt.plot(feature_count_values, accuracy_rate_by_feature_count)
    plt.xlabel("Number of principal components")
    plt.ylabel("Accuracy rate")
    plt.title("Best Accuracy Rates Across All Training Set Sizes and Neighbor Values by Number of Principal Components")
    plt.show()

    plt.plot(nearest_neighbor_values, accuracy_rate_by_num_nearest_neighbor)
    plt.xlabel("Number of nearest neighbors")
    plt.ylabel("Accuracy rate")
    plt.title("Best Accuracy Rates Across All Training Set Sizes and Counts of Principal Component by Number of Nearest Neighbors")
    plt.show()

def writeResultsToFile(k_nn_results, output_file_name):
    """
    Write the results from several executions of the k-NN classifier to a file in CSV format.

    Args:
        k_nn_results (list of KnnResults):  List of results for every execution (each has a different permutation of
                                            training set size, feature count, and k-value).
        output_file_name (string):          Name of the file to write the results to.
    """

    with open(output_file_name, 'w') as results_file:
        header_line = ["num_training_samples", "num_principal_components", "num_nearest_neighbors", "accuracy_rate",
            "classification_time_per_sample"]
        csv_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header_line)
        for result in k_nn_results:
            csv_writer.writerow([result.num_training_samples, result.num_principal_components,
                result.num_nearest_neighbors, result.accuracy_rate, result.classification_time_per_sample])

def executeExhaustiveResultsComputation(training_data, training_labels, test_data, test_labels, num_rows_in_img,
    num_cols_in_img, k_values_to_try, skip_display):

    """
    Run PCA and k-NN classifiers for many permutations of k value, feature count, and training set size.

    Args:
        training_data (2D numpy array):     N x M matrix with the training data, with M columns (one per sample) and N
                                            rows (one per pixel).
        training_labels (1D numpy array):   Labels for the training data. Has M entries.
        test_data (2D numpy array):         N x L matrix with the test data, with L columns (one per test sample) and N
                                            rows (one per pixel).
        test_labels (1D numpy array):       Labels for the test data. Has L entries.
        num_rows_in_img (int):              Number of rows in an image.
        num_cols_in_img (int):              Number of columns in an image.
        k_values_to_try (list of ints):     K values to evaluate. Defaults to a range of values between 1 and 100 if this is empty.
        skip_display (boolean):             If true, don't display plots. If false, display the plots.
    """

    img_vector_size = num_rows_in_img * num_cols_in_img

    num_training_size_increment = 50
    num_samples_to_use = [min(num_training_size_increment * i, img_vector_size)  for i in range(1, 1 + ceil(img_vector_size / num_training_size_increment))]

    test_data_subset, test_label_subset = getRandomSubsetOfData(test_data, test_labels, 1000)

    k_nn_results = []
    results_by_feature_count = {}

    for train_sample_size in num_samples_to_use:
        train_data_subset, train_label_subset = getRandomSubsetOfData(training_data, training_labels, train_sample_size)

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
            if (len(k_values_to_try) == 0):
                k_val_inc = 5
                max_single_inc_range = 15
                k_vals_to_try = [i + 1 for i in range(max_single_inc_range)]
                max_k_val = min(train_sample_size, 100)
                k_vals_to_try.extend([min(max_k_val, k_val_inc * i) for i in range((1 + floor(max_single_inc_range / k_val_inc)), 1 + ceil(max_k_val / k_val_inc))])
            else:
                k_vals_to_try = [k for k in k_values_to_try if k <= train_sample_size]
            print("K Values to Evaluate")
            print(k_vals_to_try)
            k_nearest_neighbor_classifiers = {i:KNearestNeighborClassifier(i, feature_count) for i in k_vals_to_try}
            classifiers = k_nearest_neighbor_classifiers.values() # TODO populate once have classifiers to evaluate (consider trying out other classifiers too?)

            classifier_test_info = {}
            for classifier_k_val, classifier in k_nearest_neighbor_classifiers.items():
                classifier.trainClassifier(transformed_training_data, train_label_subset)
                classifier_test_info[classifier_k_val] = classifier.testClassifier(transformed_test_data, test_label_subset)

            best_knn_k_num = 0
            best_knn_accuracy_rate = 0
            knn_results_x = k_nearest_neighbor_classifiers.keys()
            knn_results_y = []

            for k_num in k_nearest_neighbor_classifiers.keys():

                classifier_results = classifier_test_info[k_num]

                k_nn_results.append(KnnResults(train_sample_size, feature_count, k_num, classifier_results[0], classifier_results[1]))
                knn_results_y.append(classifier_results[0])

            results_by_feature_count[feature_count] = (knn_results_x, knn_results_y)

    output_file_name = "assign_1_results_" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".csv"
    writeResultsToFile(k_nn_results, output_file_name)

    if (not skip_display):
        for feature_count in num_features_to_try:
            results_for_feature_count = results_by_feature_count[feature_count]
            plt.plot(results_for_feature_count[0], results_for_feature_count[1], 'rD-.')

            plt.xlabel("K in KNN")
            plt.ylabel("Accuracy Rate")
            plt.title("Accuracy Rate by K Value for training set size " + str(train_sample_size) + " and " + str(feature_count) + " principal components")
            plt.legend()
            plt.show()

        plotResults(k_nn_results)

def evaluateAccuracyForFixedTrainingSetSize(k_val_for_classifier, training_data, training_labels, test_data,
    test_labels, skip_display):

    """
    Evaluate the accuracy of classifiers for a fixed training set size.

    Args:
        k_val_for_classifier (int):         Number of nearest neighbors to use in the classifier.
        training_data (2D numpy array):     N x M matrix with the training data, with M columns (one per sample) and N
                                            rows (one per pixel).
        training_labels (1D numpy array):   Labels for the training data. Has M entries.
        test_data (2D numpy array):         N x L matrix with the test data, with L columns (one per test sample) and N
                                            rows (one per pixel).
        test_labels (1D numpy array):       Labels for the test data. Has L entries.
        skip_display (boolean):             True if the display of plots should be skipped, false if plots should be
                                            displayed.
    """

    full_train_set_size = training_data.shape[1]
    if (k_val_for_classifier > full_train_set_size):
        print("Bad k value. Not evaluating at k=" + k_val_for_classifiers)

    # Use all training samples and vary the number of features used
    # Create a list of the different feature sizes to try out. Have more dense distribution for smaller values of
    # feature count and spread out the feature counts higher in the range (so we don't have to try out all 784
    # possible values)
    max_single_inc_range = 30
    max_small_inc_range = 75
    small_inc = 5
    large_inc = 25

    num_features_to_try = [i + 1 for i in range(max_single_inc_range) if i < full_train_set_size]
    num_features_to_try.extend([i for i in range(max_single_inc_range + small_inc, max_small_inc_range, small_inc)
        if i < full_train_set_size])
    if (full_train_set_size >= max_small_inc_range):
        num_features_to_try.extend([i for i in range(max_small_inc_range, full_train_set_size + 1, large_inc)])
        if (full_train_set_size not in num_features_to_try):
            num_features_to_try.append(full_train_set_size)

    print("Number of features to try with full training set (784 images): " + str(num_features_to_try))

    # Find the eigenvector representation from the training data
    mean_feature_vec, eigenvectors_of_train = hw1FindEigendigits(training_data)

    classifier_results = []
    for feature_count in num_features_to_try:
        print("Train set size: " + str(full_train_set_size) + ", Feature count " + str(feature_count))

        # Reduce the feature count and transform the training data to the reduced feature representation
        reduced_eigenvectors_of_train = eigenvectors_of_train[:, :feature_count]
        transformed_training_data = transformToEigenRepresentation(mean_feature_vec, reduced_eigenvectors_of_train, training_data)

        # Also transform the test data
        transformed_test_data = transformToEigenRepresentation(mean_feature_vec, reduced_eigenvectors_of_train, test_data)

        # Train and run the classifier and store the accuracy results
        classifier = KNearestNeighborClassifier(k_val_for_classifier, feature_count)
        classifier.trainClassifier(transformed_training_data, training_labels)
        classifier_results.append(classifier.testClassifier(transformed_test_data, test_labels)[0])

    print("Results (accuracy rate, num_features_to_try)")
    for i in range(len(num_features_to_try)):
        print("(" + str(classifier_results[i]) + ", " + str(num_features_to_try[i]) + ")")

    if (not skip_display):
        plotGraph(num_features_to_try, {"kNN accuracy": classifier_results}, "Number of features (eigenvectors) in classifier",
        "Accuracy rate", ("Accuracy rate by number of eigenvectors used, for " + str(full_train_set_size) +
        " samples and k=" + str(k_val_for_classifier)))

def evaluateNeuralNet(train_data, train_labels, test_data, test_labels, using_full_img):
    """
    Evaluate a neural net classifier's accuracy.

    Args:
        train_data (2D numpy array):        N x M matrix with the training data, with M columns (one per sample) and N
                                            rows (one per pixel).
        train_labels (1D numpy array):      Labels for the training data. Has M entries.
        test_data (2D numpy array):         N x L matrix with the test data, with L columns (one per test sample) and N
                                            rows (one per pixel).
        test_labels (1D numpy array):       Labels for the test data. Has L entries.
        using_full_img (boolean):           True if the neural net is being executed on the raw image data (and thus
                                            the data should be normalized), false if the neural net is working on a
                                            transformed feature representation and no additional transformation should
                                            be applied.
    Returns:
        Tuple of accuracy and average time to classify each sample.
    """

    classifier = NeuralNetClassifier()
    if (using_full_img):
        train_data = normalizeImageDataForNeuralNet(train_data)
        test_data = normalizeImageDataForNeuralNet(test_data)
    classifier.trainClassifier(train_data, train_labels)
    results = classifier.testClassifier(test_data, test_labels)
    return results

def evaluateAccuracyForFixedFeatureCount(k_val_for_classifier, feature_count_to_use, training_data, training_labels,
    test_data, test_labels, skip_display, run_neural_net):

    """
    Evaluate the accuracy for a fixed feature count and variable training set sizes.

    Args:
        k_val_for_classifier (int):         Number of nearest neighbors to use in the classifier. Should be less than
                                            the minimum number of training samples.
        feature_count_to_use (int):         Feature count to use. Should be less than the minimum number of training
                                            samples.
        training_data (2D numpy array):     N x M matrix with the training data, with M columns (one per sample) and N
                                            rows (one per pixel).
        training_labels (1D numpy array):   Labels for the training data. Has M entries.
        test_data (2D numpy array):         N x L matrix with the test data, with L columns (one per test sample) and N
                                            rows (one per pixel).
        test_labels (1D numpy array):       Labels for the test data. Has L entries.
        skip_display (boolean):             True if the display of plots should be skipped, false if plots should be
                                            displayed.
        run_neural_net (boolean):           True if the neural net classifiers should be evaluated, false if we should
                                            just use k-NN
    """
    full_train_set_size = training_data.shape[1]

    num_training_size_increment = 50
    training_set_sizes_to_try = [min(num_training_size_increment * i, full_train_set_size)  for i in range(1, 1 + ceil(full_train_set_size / num_training_size_increment))]
    training_set_sizes_to_try = [50]

    # If the feature count is greater than the smallest training set size to use, increase the feature count to be the
    # minimum value
    if (feature_count_to_use > training_set_sizes_to_try[0]):
        print("Feature count to use " + str(feature_count_to_use) + " would not work with minimum training set size " + str(training_set_sizes_to_try[0]))
        print("Changing to min training set size")
        feature_count_to_use = training_set_sizes_to_try[0]

    if (k_val_for_classifier > training_set_sizes_to_try[0]):
        print("K value " + str(k_val_for_classifier) + " would not work with minimum training set size " + str(training_set_sizes_to_try[0]))
        print("Changing to min training set size")
        k_val_for_classifier = training_set_sizes_to_try[0]

    knn_classifier_results = []
    full_feature_nn_results = []
    reduced_feature_nn_results = []
    pca_time_results = []
    knn_classification_time_results = []
    full_nn_time_results = []
    reduced_feature_nn_time_results = []

    for training_set_size in training_set_sizes_to_try:
        print("Train set size: " + str(training_set_size) + ", Feature count " + str(feature_count_to_use))

        # Get the subset of the training data and compute the eigenvectors and mean for that training set
        train_data_subset, train_label_subset = getRandomSubsetOfData(training_data, training_labels, training_set_size)
        principal_component_start_time = time.time()
        mean_feature_vec, eigenvectors_of_train = hw1FindEigendigits(train_data_subset)
        principal_component_extraction_duration = time.time() - principal_component_start_time
        pca_time_results.append(principal_component_extraction_duration)

        # Get the eigenvectors to use in the feature transformation
        reduced_eigenvectors_of_train = eigenvectors_of_train[:, :feature_count_to_use]

        # Transform the training and test data
        transformed_training_data = transformToEigenRepresentation(mean_feature_vec, reduced_eigenvectors_of_train, train_data_subset)
        transformed_test_data = transformToEigenRepresentation(mean_feature_vec, reduced_eigenvectors_of_train, test_data)

        # Train and run the classifier and store the accuracy results
        classifier = KNearestNeighborClassifier(k_val_for_classifier, feature_count_to_use)
        classifier.trainClassifier(transformed_training_data, train_label_subset)
        knn_result = classifier.testClassifier(transformed_test_data, test_labels)
        knn_classifier_results.append(knn_result[0])
        knn_classification_time_results.append(knn_result[1])

        if (run_neural_net):
            print("Testing full feature nn")
            full_feature_nn_result = evaluateNeuralNet(train_data_subset, train_label_subset, test_data, test_labels, True)
            full_feature_nn_results.append(full_feature_nn_result[0])
            full_nn_time_results.append(full_feature_nn_result[1])
            print("Testing 25 feature nn")
            reduced_feature_nn_result = evaluateNeuralNet(transformed_training_data, train_label_subset, transformed_test_data, test_labels, False)
            reduced_feature_nn_results.append(reduced_feature_nn_result[0])
            reduced_feature_nn_time_results.append(reduced_feature_nn_result[1])

    if (run_neural_net):
        print("Results (KNN accuracy rate, full feature neural net accuracy rate, limited feature neural net accuracy rate, training sample size)")
        for i in range(len(training_set_sizes_to_try)):
            print("(" + (','.join(map(str, [knn_classifier_results[i], full_feature_nn_results[i], reduced_feature_nn_results[i], training_set_sizes_to_try[i]]))) + ")")
        print("Results (KNN time, full feature neural net time, limited feature neural net time, training sample size)")
        for i in range(len(training_set_sizes_to_try)):
            print("(" + (','.join(map(str, [knn_classification_time_results[i], full_nn_time_results[i], reduced_feature_nn_time_results[i], training_set_sizes_to_try[i]]))) + ")")
    else:
        print("Results (KNN accuracy rate, training sample size)")
        for i in range(len(training_set_sizes_to_try)):
            print("(" + (','.join(map(str, [knn_classifier_results[i], training_set_sizes_to_try[i]]))) + ")")
        print("Results (KNN time, training sample size)")
        for i in range(len(training_set_sizes_to_try)):
            print("(" + (','.join(map(str, [knn_classification_time_results[i], training_set_sizes_to_try[i]]))) + ")")

    if (not skip_display):
        if (run_neural_net):
            accuracy_y_vals = {"kNN": knn_classifier_results, "Original Image Neural Net": full_feature_nn_results, "Principal Components Neural Net": reduced_feature_nn_results}
            plotGraph(training_set_sizes_to_try, accuracy_y_vals, "Number of training samples",
            "Accuracy rate", ("Accuracy rate by number of training samples used, for feature count " + str(feature_count_to_use) +
            " and k=" + str(k_val_for_classifier)),
            {"kNN": 'bD-', "Original Image Neural Net": 'gD-', "Principal Components Neural Net": 'mD-'})
            time_y_vals = {"kNN": knn_classification_time_results, "Original Image Neural Net": full_nn_time_results, "Principal Components Neural Net": reduced_feature_nn_time_results}
            plotGraph(training_set_sizes_to_try, time_y_vals, "Number of training samples", "Avg time per test sample",
                ("Avg Classification Time Per Test Sample for feature count " + str(feature_count_to_use) +
                " and k=" + str(k_val_for_classifier)), {"kNN": 'bD-', "Original Image Neural Net": 'gD-', "Principal Components Neural Net": 'mD-'})
        else:
            accuracy_y_vals = {"kNN": knn_classifier_results}
            plotGraph(training_set_sizes_to_try, accuracy_y_vals, "Number of training samples",
            "Accuracy rate", ("Accuracy rate by number of training samples used, for feature count " + str(feature_count_to_use) +
            " and k=" + str(k_val_for_classifier)), {"kNN": 'bD-'})
            time_y_vals = {"kNN": knn_classification_time_results}
            plotGraph(training_set_sizes_to_try, time_y_vals, "Number of training samples", "Avg time per test sample",
                ("Avg Classification Time Per Test Sample for feature count " + str(feature_count_to_use) +
                " and k=" + str(k_val_for_classifier)), {"kNN": 'bD-'})
        plotGraph(training_set_sizes_to_try, {"PCA time": pca_time_results}, "Number of training samples",
            "Time to compute principal component vectors", "Time to compute principal components")

def displayResultsOfPcaAndClassification(num_rows_in_img, num_cols_in_img, feature_count_to_use, k_val_for_classifier,
    training_data, training_labels, test_data, test_labels):
    """
    Display the results of the principal component analysis and classification for a handful of test images.
    This is just a demonstration of the techniques and not an analysis of the performance.

    Args:
        num_rows_in_img (int):              Number of rows in images.
        num_cols_in_img (int):              Number of columns in images.
        feature_count_to_use (int):         Feature count to use. Should be less than the minimum number of training
                                            samples.
        k_val_for_classifier (int):         Number of nearest neighbors to use in the classifier. Should be less than
                                            the minimum number of training samples.
        training_data (2D numpy array):     N x M matrix with the training data, with M columns (one per sample) and N
                                            rows (one per pixel).
        training_labels (1D numpy array):   Labels for the training data. Has M entries.
        test_data (2D numpy array):         N x L matrix with the test data, with L columns (one per test sample) and N
                                            rows (one per pixel).
        test_labels (1D numpy array):       Labels for the test data. Has L entries.
    """
    mean_feature_vec, eigenvectors_of_train = hw1FindEigendigits(training_data)
    reduced_eigenvectors_of_train = eigenvectors_of_train[:, :feature_count_to_use]
    transformed_training_data_small_feature_count = transformToEigenRepresentation(mean_feature_vec,
        reduced_eigenvectors_of_train, training_data)
    transformed_training_data_full_feature_count = transformToEigenRepresentation(mean_feature_vec,
        eigenvectors_of_train, training_data)

    num_eigen_vecs_to_display = 10
    for i in range(num_eigen_vecs_to_display):
        eigen_vec = eigenvectors_of_train[:, i]
        displayEigenVector(num_rows_in_img, num_cols_in_img, eigen_vec)

    classifier = KNearestNeighborClassifier(k_val_for_classifier, feature_count_to_use)
    classifier.trainClassifier(transformed_training_data_small_feature_count, training_labels)

    num_test_samples_to_classify_and_display = 10
    rand_test_imgs_to_display, rand_test_labels_to_display = getRandomSubsetOfData(test_data, test_labels,
        num_test_samples_to_classify_and_display)
    transformed_test_imgs_to_display_small_feature_count = transformToEigenRepresentation(mean_feature_vec,
        reduced_eigenvectors_of_train, rand_test_imgs_to_display)
    transformed_test_imgs_to_display_full_feature_count = transformToEigenRepresentation(mean_feature_vec,
        eigenvectors_of_train, rand_test_imgs_to_display)
    for i in range(num_test_samples_to_classify_and_display):
        img_vector = rand_test_imgs_to_display[:, i]
        transformed_img_vector_small_feature_count = transformed_test_imgs_to_display_small_feature_count[:, i]
        transformed_img_vector_full_feature_count = transformed_test_imgs_to_display_full_feature_count[:, i]
        classifier_label = classifier.classifySample(transformed_img_vector_small_feature_count)
        image_matrix = convertVectorToImage(img_vector, num_rows_in_img, num_cols_in_img)
        displayImage(image_matrix, rand_test_labels_to_display[i], classifier_label)

        displayEigenVectorRepresentationOfImage(transformed_img_vector_small_feature_count,
            reduced_eigenvectors_of_train, num_rows_in_img, num_cols_in_img, mean_feature_vec)
        displayEigenVectorRepresentationOfImage(transformed_img_vector_full_feature_count, eigenvectors_of_train,
            num_rows_in_img, num_cols_in_img, mean_feature_vec)

def executeSimpleEvaluation(training_data, training_labels, test_data, test_labels, num_rows_in_img, num_cols_in_img,
    k_values_to_try, skip_display, run_neural_net):

    """
    Execute a simple evaluation that computes accuracy for one fixed feature count and a variety of training set sizes,
    and also one fixed training set size and a variety of feature count numbers. Also displays some eigenvectors as
    images, some images reconstructed from their eigenvector representation, and the result of classification for a few
    images.

    Args:
        training_data (2D numpy array):     N x M matrix with the training data, with M columns (one per sample) and N
                                            rows (one per pixel).
        training_labels (1D numpy array):   Labels for the training data. Has M entries.
        test_data (2D numpy array):         N x L matrix with the test data, with L columns (one per test sample) and N
                                            rows (one per pixel).
        test_labels (1D numpy array):       Labels for the test data. Has L entries.
        num_rows_in_img (int):              Number of rows in images.
        num_cols_in_img (int):              Number of columns in images.
        k_values_to_try (list of ints):     List of k values to try in K nearest neighbors classifiers. If empty, will
                                            use a single default value. If multiple entries, the functionality
                                            demonstrated will be executed once per k value.
        skip_display (boolean):             True if the display of plots should be skipped, false if plots should be
                                            displayed.
        run_neural_net (boolean):           True if the neural net classifiers should be evaluated, false if we should
                                            just use k-NN
    """

    if (len(k_values_to_try) == 0):
        k_values_to_try = [10] # TODO come up with good value

    # Display a handful of training images and test images
    num_train_to_display = 3
    num_test_to_display = 3

    img_vector_size = num_rows_in_img * num_cols_in_img

    print("Displaying some random training data and test data")
    rand_train_imgs_to_display, rand_train_labels_to_display = getRandomSubsetOfData(training_data, training_labels,
        num_train_to_display)

    # Display some images from both the training and test data sets
    if (not skip_display):
        for i in range(num_train_to_display):
            img_vector = rand_train_imgs_to_display[:, i]
            image_matrix = convertVectorToImage(img_vector, num_rows_in_img, num_cols_in_img)
            displayImage(image_matrix, rand_train_labels_to_display[i])

        rand_test_imgs_to_display, rand_test_labels_to_display = getRandomSubsetOfData(test_data, test_labels,
            num_test_to_display)
        for i in range(num_test_to_display):
            img_vector = rand_test_imgs_to_display[:, i]
            image_matrix = convertVectorToImage(img_vector, num_rows_in_img, num_cols_in_img)
            displayImage(image_matrix, rand_test_labels_to_display[i])

    # Get subset of test data for faster evaluation
    test_set_size_to_use = 500
    test_data_subset, test_label_subset = getRandomSubsetOfData(test_data, test_labels, test_set_size_to_use)

    # Get a subset of data from the training set that is the same as the number of pixels
    train_data_subset, train_label_subset = getRandomSubsetOfData(training_data, training_labels, img_vector_size)

    for k_val in k_values_to_try:

        # Use all training samples and vary the number of features used
        # Create a list of the different feature sizes to try out. Have more dense distribution for smaller values of
        # feature count and spread out the feature counts higher in the range (so we don't have to try out all 784
        # possible values)
        evaluateAccuracyForFixedTrainingSetSize(k_val, train_data_subset, train_label_subset, test_data_subset,
            test_label_subset, skip_display)

        # Using a fixed feature count, vary the number of training samples used
        # Training sample count must be greater than or equal to the fixed feature count
        feature_count_to_use = 25 # TODO
        evaluateAccuracyForFixedFeatureCount(k_val, feature_count_to_use, train_data_subset, train_label_subset,
            test_data, test_labels, skip_display, run_neural_net)

        # Display the eigenvectors
        # Display some images transformed to the eigenvector representation
        # Display a handful of images and their ground truth and classifier-produced labels
        if not skip_display:
            displayResultsOfPcaAndClassification(num_rows_in_img, num_cols_in_img, feature_count_to_use, k_val,
                train_data_subset, train_label_subset, test_data_subset, test_label_subset)

def getCmdLineArgs():

    """
    Set up the command line argument options and return the arguments used.

    Returns:
        Command line arguments data structure.
    """
    default_training_data_file_name = "/Users/mandiadkins/Downloads/train-images.idx3-ubyte"
    default_training_label_file_name = "/Users/mandiadkins/Downloads/train-labels.idx1-ubyte"

    default_test_data_file_name = "/Users/mandiadkins/Downloads/t10k-images.idx3-ubyte"
    default_test_label_file_name = "/Users/mandiadkins/Downloads/t10k-labels.idx1-ubyte"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exhaustive", action='store_true')
    arg_parser.add_argument("--train_image_file", default=default_training_data_file_name, help='Name of the file '\
        'containing the training images. Should be formatted based on the MNIST format described on the MNIST website.')
    arg_parser.add_argument("--train_label_file", default=default_training_label_file_name, help='Name of the file '\
        'containing the training labels. Should be formatted based on the MNIST format described on the MNIST website.')
    arg_parser.add_argument("--test_image_file", default=default_test_data_file_name, help='Name of the file '\
        'containing the test images. Should be formatted based on the MNIST format described on the MNIST website.')
    arg_parser.add_argument("--test_label_file", default=default_test_label_file_name, help='Name of the file'\
        'containing the test labels. Should be formatted based on the MNIST format described on the MNIST website.')
    arg_parser.add_argument("--knn_k_value", "-k", type=int, nargs='+', help='List of k values to try in program'\
        'execution. If in normal mode, if this is not specified, a default k value will be used. If multiple are '\
        'specified, the accuracy and classification results will be run per k value. If in exhaustive mode and no k '\
        'values are specified, a large array of k values will be tested')
    arg_parser.add_argument("--skip_display", "-s", action='store_true',
        help='This option should be specified to skip the plots and image display.')
    arg_parser.add_argument("--neural-net", "-n", action='store_true',
        help='This option should be specified to run the neural net classifiers in the simple evaluation')

    return arg_parser.parse_args()

if __name__=="__main__":

    parser_results = getCmdLineArgs()

    run_exhaustive_results = parser_results.exhaustive
    train_image_file_name = parser_results.train_image_file
    train_label_file_name = parser_results.train_label_file
    test_image_file_name = parser_results.test_image_file
    test_label_file_name = parser_results.test_label_file
    k_values_to_try = parser_results.knn_k_value
    skip_display = parser_results.skip_display
    run_neural_net = parser_results.neural_net

    # Load the training data
    training_data, num_rows_in_img, num_cols_in_img = readMNISTData(train_image_file_name)
    training_labels = readMNISTLabels(train_label_file_name)

    # Load the test data
    test_data, num_rows_in_img, num_cols_in_img = readMNISTData(test_image_file_name)
    test_labels = readMNISTLabels(test_label_file_name)

    # Transpose the training and test data so that each image is a column
    training_data = np.transpose(training_data)
    test_data = np.transpose(test_data)

    # If no k values were supplied, switch to empty list to indicate use of default
    if (k_values_to_try == None):
        k_values_to_try = []

    if (run_exhaustive_results):
        print("Running exhaustive results calcluation")
        executeExhaustiveResultsComputation(training_data, training_labels, test_data, test_labels, num_rows_in_img,
            num_cols_in_img, k_values_to_try, skip_display)

    else:
        print("Running simple evaluation")
        executeSimpleEvaluation(training_data, training_labels, test_data, test_labels, num_rows_in_img,
            num_cols_in_img, k_values_to_try, skip_display, run_neural_net)
