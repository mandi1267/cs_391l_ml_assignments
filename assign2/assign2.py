"""
Amanda Adkins
Fall 2020 CS391L Assignment 2
"""


import numpy as np
from scipy.special import expit as sigmoid
import argparse
import scipy.io
import matplotlib.pyplot as plt
import scipy.signal
import datetime

# import joblib

class ICARunConfiguration(object):
    """
    Configuration for a run of ICA.
    """

    def __init__(self, mixing_matrix, original_signals, max_iterations, convergence_threshold, learning_rate):
        """
        Create the run configuration with any pieces of data needed to run ICA.

        Args:
            mixing_matrix (2D Numpy Array):     Array used to mix the original signals to get the mixed signals.
            original_signals (2D Numpy Array):  Array of original signals, with each signal as a row.
            max_iterations (int):               Maximum number of iterations that can be used to run ICA.
            convergence_threshold (double):     Convergence threshold. Value that each value in the gradient of the W
                                                estimate must be under in order to be considered converged.
            learning_rate (double):             Learning rate to use in gradient descent for ICA. Controls rate of
                                                descent.
        """
        self.mixing_matrix = mixing_matrix
        self.original_signals = original_signals
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.learning_rate = learning_rate

class ICAResults(object):
    """
    Results for a run of ICA.
    """

    def __init__(self, unmixing_matrix, unmixed_sounds, number_of_iterations, correlation_indices, correlation_values):

        """
        Create the results object that keeps track of all information related to ICA that is not known when ICA is
        started.

        Args:
            unmixing_matrix (2D Numpy Array):       Array found that extracts the original signals from the mixed
                                                    signals.
            unmixed_sounds (2D Numpy Array):        Array of recovered signals, with each signal as a row.
            number_of_iterations (int):             Number of iterations run. Will either be until convergence or until
                                                    the maximum number of configurations is reached.
            correlation_indices (1D Numpy Array):   Correlation indices. The nth entry gives the index of the signal in
                                                    the unmixed sounds matrix that corresponds to the nth original
                                                    signal.
            correlation_values (1D Numpy Array):    Correlation values. The nth entry gives the correlation result of
                                                    the nth original signal to the matching signal (noted in
                                                    correlation-indices) from the recovered signal matrix.
        """
        self.unmixing_matrix = unmixing_matrix
        self.unmixed_sounds = unmixed_sounds
        self.number_of_iterations = number_of_iterations
        self.correlation_indices = correlation_indices
        self.correlation_values = correlation_values


class ICARunData(object):
    """
    Full run data (configuration and results) for a run of ICA).
    """

    def __init__(self, ica_run_configuration, ica_results):
        """
        Create the run data object.

        Args:
            ica_run_configuration (ICARunConfiguration):    Configuration for the ICA run.
            ica_results (ICAResults):                       Results for the ICA run.
        """
        self.ica_run_configuration = ica_run_configuration
        self.ica_results = ica_results


def readSoundData(sound_file_name):
    """
    Read the sound data from the given file.

    Args:
        sound_file_name (string): Name of the file from which to read the original full data set of sounds.
    Returns:
        2D Numpy array with each read signal as a different row of the matrix.
    """

    data_in_file = scipy.io.loadmat(sound_file_name)
    sounds_mat_key = "sounds"
    sounds_mat = data_in_file[sounds_mat_key]
    return sounds_mat

def readTestData(test_data_file_name):
    """
    Read the small data set and the mixing matrix from the file.

    Args:
        test_data_file_name (string): Name of the file from which to read the small data set and mixing matrix.
    Returns:
        Tuple of the 2D Numpy array representing the mixing matrix and 2D Numpy array representing the smaller data set
        of signals.
    """
    data_in_file = scipy.io.loadmat(test_data_file_name)
    a_mat_key = "A"
    u_mat_key = "U"
    return (data_in_file[a_mat_key], data_in_file[u_mat_key])

def createRandomSoundMixingMatrixOfSize(mat_size):
    """
    Create a random matrix for mixing sounds.

    Args:
        mat_size (tuple): Two element tuple giving the dimensions of the mixing matrix to randomly initialize.
    Returns:
        2D Numpy array to use as a mixing matrix.
    """
    return np.random.uniform(0, 1, mat_size)

def plotSignals(original_signals, mixed_signals, extracted_signals, title, correlation_indices=None, correlation_values=None):
    """
    Plot the original, mixed, and recovered signals and if provided, display the correlation results of the recovered signals.

    Args:
        original_signals (2D Numpy Array):      Signals that were mixed.
        mixed_signals (2D Numpy Array):         Mixed signals from which the original signals were reconstructed.
        extracted_signals (2D Numpy Array):     Signals that should resemble the original signals that were recovered
                                                from the mixed signals.
        title (string):                         Title to display on the chart
        correlation_indices (1D Numpy Array):   Correlation indices. The nth entry gives the index of the signal in
                                                the unmixed sounds matrix that corresponds to the nth original
                                                signal.
        correlation_values (1D Numpy Array):    Correlation values. The nth entry gives the correlation result of
                                                the nth original signal to the matching signal (noted in
                                                correlation-indices) from the recovered signal matrix.
    """
    num_original_signals = original_signals.shape[0]
    num_mixed_signals = mixed_signals.shape[0]
    num_extracted_signals = extracted_signals.shape[0]
    signal_len = original_signals.shape[1]
    normalized_mixed = normalizeSoundMatrixRows(mixed_signals)
    normalized_original = normalizeSoundMatrixRows(original_signals)

    display_correlations = not ((correlation_indices is None) or (correlation_values is None))

    fig, ax = plt.subplots(1)
    num_plotted_signals = 0
    for i in range(num_original_signals):
        if (not display_correlations):
            plt.plot(normalized_original[i] + num_plotted_signals)
        else:
            plot_label = "Orig. signal " + str(i)
            plt.plot(normalized_original[i] + num_plotted_signals, label=plot_label)
        num_plotted_signals += 1
    for i in range(num_extracted_signals):
        if (not display_correlations):
            plt.plot(extracted_signals[i] + num_plotted_signals)
        else:
            original_signal_index = correlation_indices.index(i)
            correlation_value = correlation_values[original_signal_index]
            corr_label = "Extracted version of " + str(original_signal_index) + ", correlation: " + str(correlation_value)
            plt.plot(extracted_signals[i] + num_plotted_signals, label=corr_label)
        num_plotted_signals += 1
    for i in range(num_mixed_signals):
        plt.plot(normalized_mixed[i] + num_plotted_signals)
        num_plotted_signals += 1
    left, width = -2, .5
    bottom, height = 0, (num_extracted_signals + num_mixed_signals + num_original_signals)


    plt.text(left, 0.5 * height, 'Recovered',
            horizontalalignment='right',
            verticalalignment='center',
            rotation='vertical')
    plt.text(left, 0.15 * height, 'Original',
             horizontalalignment='right',
             verticalalignment='center',
             rotation='vertical')
    plt.text(left, 0.85 * height, 'Mixed',
             horizontalalignment='right',
             verticalalignment='center',
             rotation='vertical')
    ax.set_yticklabels([])
    box = ax.get_position()
    if (display_correlations):
        ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75])
        fig.legend(loc='lower center', ncol=2)
    plt.title(title)
    plt.hlines([num_original_signals, (num_extracted_signals + num_original_signals)], xmin=0, xmax=signal_len)
    plt.show()

def initializeUnmixingMatrix(num_rows, num_columns):
    """
    Initialize the unmixing matrix W.

    Args:
        num_rows (integer):     Number of rows that the matrix should have.
        num_columns (integer):  Number of columns that the matrix should have.
    Returns:
        Initialized value for the unmixing matrix.
    """
    return np.random.uniform(0.0, 0.1, (num_rows, num_columns))

def calculateZMat(unmixed_estimate):
    """
    Calculate the matrix Z, for which each entry is the sigmoid function evaluated at each entry in the given unmixed
    sound estimate.

    Args:
        unmixed_estimate (2D Numpy Array):  Estimate of the unmixed signals. Should take the sigmoid function of this
                                            matrix element-wise to get the matrix Z.
    Returns:
        2D Numpy array that is the element-wise evaluation of the sigmoid function for the matrix Y (current
        estimate of unmixed signals).
    """
    return sigmoid(unmixed_estimate)

def calculateGradientW(learning_rate, unmixed_estimate_y, z_mat, unmixing_matrix_w):

    """
    Calculate the gradient of the unmixing matrix W.

    Args:
        learning_rate (double):                 Learning rate to use in computing the gradient.
        unmixed_estimate_y (2D Numpy array):    Current estimate of the unmixed signals.
        z_mat (2D Numpy array):                 Matrix z that contains the element-wise sigmoid value of the unmixed
                                                estimate Y.
        unmixing_matrix_w (2D Numpy array):     Current estimate of the unmixing matrix W that will recover the
                                                original signals from the mixed signals.
    Returns:
        Gradient of W.
    """

    multiplied_identity = np.identity(unmixing_matrix_w.shape[0]) * unmixed_estimate_y.shape[1]

    grad_w = learning_rate * (multiplied_identity + ((1 - (2 * z_mat)) @ np.transpose(unmixed_estimate_y))) @ unmixing_matrix_w
    return grad_w

def checkForConvergence(matrix_diff_grad_w, convergence_threshold):
    """
    Check the gradient for convergence. Returns true if the absolute value of every entry is less than the convergence
    threshold.

    Args:
        matrix_diff_grad_w (2D Numpy array):    Gradient of w to use for the convergence check.
        convergence_threshold (double):         Value which each entry in the gradient must be under for the algorithm
                                                to be considered converged.
    Returns:
        True if the gradient is sufficiently small and thus the algorithm has converged, false otherwise.
    """

    abs_mat_diff = abs(matrix_diff_grad_w)
    return (abs_mat_diff < convergence_threshold).all()

def normalizeSoundMatrixRows(sound_matrix):
    """
    Normalize the sound matrix so that its values are between 0 and 1 inclusive. Normalization is done per row, so each
    row has a max of 1 and min of 0.

    Args:
        sound_matrix (2D Numpy array):  Matrix of sounds to normalize.
    Returns:
        Normalized version of the sound matrix.
    """
    max_per_row = np.amax(sound_matrix, axis=1, keepdims=True)
    min_per_row = np.amin(sound_matrix, axis=1, keepdims=True)

    range_per_row = max_per_row - min_per_row
    min_subtracted_mat = sound_matrix - min_per_row
    normalized_mat = min_subtracted_mat / range_per_row
    return normalized_mat

def computeOptimalSignalCorrelations(original_signals, extracted_signals):
    """
    Compute the optimal signal correlation pairings. Finds the extracted signal that best matches each original signal,
    with each extracted signal only being used once.

    Args:
        original_signals (2D Numpy Array):  Matrix of original signals.
        extracted_signals (2D Numpy Array): Matrix of recovered signals.
    Returns:
        Tuple of optimal correlation indices and optimal correlation values. Optimal correlation indices is a 1D array
        whose nth entry provides the index of the recovered signal that best aligns with the nth original signal.
        Optimal correlation values is a 1D array whose nth entry provides the correlation between the matched recovered
        signal and the nth original signal.
    """

    optimal_correlation_indices = []
    optimal_correlations = []
    for i in range(original_signals.shape[0]):
        optimal_correlation = 0
        optimal_correlation_index = -1
        for j in range(extracted_signals.shape[0]):
            correlation = np.corrcoef(original_signals[i], extracted_signals[j])[0, 1]
            if not (j in optimal_correlation_indices):
                if abs(correlation) > abs(optimal_correlation):
                    optimal_correlation_index = j
                    optimal_correlation = correlation

        optimal_correlation_indices.append(optimal_correlation_index)
        optimal_correlations.append(optimal_correlation)

    return optimal_correlation_indices, optimal_correlations

def extractOriginalSounds(mixed_sounds, learning_rate, max_iterations, num_original_sounds, convergence_threshold):
    """
    Extract the original sounds from the mixed sounds using the given parameters.

    Args:
        mixed_sounds (2D Numpy Array):  Mixed sounds matrix.
        learning_rate (double):         Learning rate to use in gradient descent.
        max_iterations (integer):       Maximum number of iterations that gradient descent can run.
        num_original_sounds (integer):  Number of original sounds, and consequently, number of sounds to recover.
        convergence_threshold (double): Threshold for the gradient for the algorithm to be considered converged.
    Returns:
        3-element tuple of unmixing matrix, recovered sounds, and number of iterations executed (either until
        convergence or until max iterations was hit).
    """

    # Step 2 of Gradient Descent Method
    unmixing_matrix_w = initializeUnmixingMatrix(num_original_sounds, mixed_sounds.shape[0])

    converged = False

    iterations = 0
    while ((not converged) and (max_iterations > iterations)):
        # Step 3 of Gradient Descent Method
        unmixed_estimate_y = unmixing_matrix_w @ mixed_sounds
        # Step 4
        z_mat = calculateZMat(unmixed_estimate_y)
        # Step 5
        gradient_w = calculateGradientW(learning_rate, unmixed_estimate_y, z_mat, unmixing_matrix_w)
        # Step 6
        # check for convergence
        converged = checkForConvergence(gradient_w, convergence_threshold)
        unmixing_matrix_w = unmixing_matrix_w + gradient_w
        iterations += 1
        if (iterations % 1000 == 0):
            print("Iterations " + str(iterations))

    approx_unmixed_sounds = normalizeSoundMatrixRows(unmixing_matrix_w @ mixed_sounds)

    return unmixing_matrix_w, approx_unmixed_sounds, iterations


def resultsForSingleRun(ica_run_configuration):
    """
    Execute a single run configuration of ICA and get the results.

    Args:
        ica_run_configuration (ICARunConfiguration): Contains all parameters and data needed to run ICA
    Returns:
        ICA Run Data object containing the results (unmixing matrix, number of iterations, and correlation data)
        along with the configuration.
    """

    mixed_sounds = ica_run_configuration.mixing_matrix @ ica_run_configuration.original_signals
    unmixing_matrix, unmixed_sounds, num_iterations = extractOriginalSounds(
            mixed_sounds, ica_run_configuration.learning_rate, ica_run_configuration.max_iterations,
            ica_run_configuration.mixing_matrix.shape[1], ica_run_configuration.convergence_threshold)

    correlation_indices, correlation_values = computeOptimalSignalCorrelations(ica_run_configuration.original_signals, unmixed_sounds)
    ica_results = ICAResults(unmixing_matrix, unmixed_sounds, num_iterations, correlation_indices, correlation_values)

    return ICARunData(ica_run_configuration, ica_results)


def getCmdLineArgs():
    """
    Get the command line arguments.

    Returns:
        Arguments parsed from the command line input.
    """

    default_sound_file_name = "/Users/mandiadkins/Downloads/sounds.mat"
    default_small_dataset_file_name = "/Users/mandiadkins/Downloads/icaTest.mat"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--sound_file_name", default=default_sound_file_name, help='Name of the file '\
        'containing the sound clips to mix and unmix.')
    arg_parser.add_argument("--small_dataset_file_name", default=default_small_dataset_file_name, help='Name of the '\
        'containing the small data set and mixing matrix.')

    return arg_parser.parse_args()


def extractSignalsAndPlot(ica_run_configuration, title):
    """
    Recover the signals using the given run configuration and plot the signals.

    Args:
        ica_run_configuration (ICARunConfiguration):    Contains parameters and data using which to run ICA.
        title (string):                                 Title of the plot to display.
    """
    ica_data = resultsForSingleRun(ica_run_configuration)
    mixed_signals = ica_run_configuration.mixing_matrix @ ica_run_configuration.original_signals
    plotSignals(ica_data.ica_run_configuration.original_signals, mixed_signals, ica_data.ica_results.unmixed_sounds,
                title, ica_data.ica_results.correlation_indices, ica_data.ica_results.correlation_values)


def plotSmallSetAndLargeSetWithDifferentAValues(large_data_set, small_dataset_file_name):
    """
    Plot the original, recovered, and mixed signals for the small data set (using the provided mixing matrix) and the
    large data set using randomized mixing matrices of different sizes.

    Args:
        large_data_set (2D Numpy array):    Large data set, which is made up of multiple signals, with each row as a
                                            signal.
        small_dataset_file_name (string):   Name of the file containing the small data set and the relevant mixing
                                            matrix.
    """

    max_iterations = 100000

    small_dataset_learning_rate = 0.01
    large_dataset_learning_rate = 0.00001

    convergence_threshold = 0.000001

    # Get small data set and A matrix
    # Recover signals and plot
    small_mixing_mat, small_original_sounds = readTestData(small_dataset_file_name)

    small_dataset_run_configuration = ICARunConfiguration(small_mixing_mat, small_original_sounds, max_iterations,
                                                          convergence_threshold, small_dataset_learning_rate)
    extractSignalsAndPlot(small_dataset_run_configuration, "Small Dataset Original, Recovered, and Mixed Signals")

    num_signals = large_data_set.shape[0]

    # Plot for the larger data set using the same number of mixed signals as there are original signals
    mat_shape = (num_signals, num_signals)
    equal_mixed_to_orig_signal_mixing_mat = createRandomSoundMixingMatrixOfSize(mat_shape)
    equal_mixed_to_orig_signal_run_config = ICARunConfiguration(equal_mixed_to_orig_signal_mixing_mat,
            large_data_set, max_iterations, convergence_threshold, large_dataset_learning_rate)
    extractSignalsAndPlot(equal_mixed_to_orig_signal_run_config, "Large Dataset Signals using " + str(num_signals) +
                          "x" + str(num_signals) + " Mixing Matrix")

    # Plot for the larger data set using the fewer mixed signals than original signals
    num_mixed_signals = num_signals - 2
    mat_shape = (num_mixed_signals, num_signals)
    less_mixed_than_orig_signal_mixing_mat = createRandomSoundMixingMatrixOfSize(mat_shape)
    less_mixed_than_orig_signal_run_config = ICARunConfiguration(less_mixed_than_orig_signal_mixing_mat,
                large_data_set, max_iterations, convergence_threshold, large_dataset_learning_rate)
    extractSignalsAndPlot(less_mixed_than_orig_signal_run_config, "Large Dataset Signals using " +
                          str(num_mixed_signals) + "x" + str(num_signals) + " Mixing Matrix")

    # Plot for the larger data set using the more mixed signals than original signals
    num_mixed_signals = num_signals + 2
    mat_shape = (num_mixed_signals, num_signals)
    more_mixed_than_orig_signal_mixing_mat = createRandomSoundMixingMatrixOfSize(mat_shape)
    more_mixed_than_orig_signal_run_config = ICARunConfiguration(more_mixed_than_orig_signal_mixing_mat,
                large_data_set, max_iterations, convergence_threshold, large_dataset_learning_rate)
    extractSignalsAndPlot(more_mixed_than_orig_signal_run_config, "Large Dataset Signals using " +
                          str(num_mixed_signals) + "x" + str(num_signals) + " Mixing Matrix")

def plotGraph(x_values, y_values_dict, x_axis_label, y_axis_label, graph_title, log_scale_x=False):
    """
    Plot a graph with the given data.

    Args:
        x_values (1D array/dict of str to 1D array):        X values to plot. Either a list of x values, or a dictionary
                                                            mapping the y label to the x values corresponding to that y
                                                            label.
        y_values (dict of string to 1D array):              Dictionary of y-value label to show in legend to the
                                                            corresponding y-values to plot. These will be plotted
                                                            against x_values.
        x_axis_label (string):                              Label for the x axis
        y_axis_label (string):                              Label for the y axis
        graph_title (string):                               Title for the graph
    """
    using_x_val_dict = type(x_values) is dict
    for label, y_values in y_values_dict.items():
        if (using_x_val_dict):
            plt.plot(x_values[label], y_values, label = label)
        else:
            plt.plot(x_values, y_values, label = label)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    # plt.ylim(top=1)
    if (log_scale_x):
        plt.xscale("log")
    plt.yscale("linear")
    plt.title(graph_title)
    if (len(y_values_dict) > 1):
        plt.legend()
    plt.show()


def getAverageIterationsForResults(run_data):
    """
    Get the average number of iterations that the given runs needed to converge (or reach max iterations).

    Args:
        run_data (list of ICARunData objects):  Contains details about runs to consider.
    Returns:
        Average number of iterations needed in the given runs.
    """
    iterations_sum = 0
    for run_info in run_data:
        iterations_sum += run_info.ica_results.number_of_iterations
    return iterations_sum / len(run_data)

def getAverageMinimumAndAverageAverageCorrelationValueForResults(run_data):
    """
    Get the correlation results for the given runs. Finds the minimum and average correlation value for each run and
     averages them over all given runs to create the averaged minimum and averaged average.

    Args:
        run_data (list of ICARunData objects):  Contains details about runs to consider.
    Returns:
        Tuple of averaged minimum correlation and averaged average correlation.
    """
    average_min_sum = 0
    average_average_sum = 0
    for run_info in run_data:
        min_correlation = 1
        correlation_sum = 0
        for correlation in run_info.ica_results.correlation_values:
            correlation_sum += abs(correlation)
            min_correlation = min(min_correlation, abs(correlation))

        average_min_sum += min_correlation
        average_average_sum += (correlation_sum / len(run_info.ica_results.correlation_values))

    return (average_min_sum / len(run_data)), (average_average_sum / len(run_data))

def displayConvergenceThresholdExperimentData(convergence_values, five_row_results_by_convergence_value, seven_row_results_by_convergence_value):
    """
    Display convergence threshold experiment result data:

    Args:
        convergence_values (1D Numpy Array):    Convergence values to display results for.
        five_row_results_by_convergence_value (dict with double keys and ICARunData lists as values):   Results for
            mixing matrices that generate 5 mixed signals, stored by the convergence threshold for which they were
            executed.
        seven_row_results_by_convergence_value (dict with double keys and ICARunData lists as values):  Results for
            mixing matrices that generate 7 mixed signals, stored by the convergence threshold for which they were
            executed.
    """

    # Get average iterations for each convergence value for 5 row mixing mats and 7 row mixing mats
    average_iter_five_row_mat = [getAverageIterationsForResults(five_row_results_by_convergence_value[convergence_value]) for convergence_value in convergence_values]
    average_iter_seven_row_mat = [getAverageIterationsForResults(seven_row_results_by_convergence_value[convergence_value]) for convergence_value in convergence_values]

    five_row_label = "5 mixed signals"
    seven_row_label = "7 mixed signals"

    plotGraph(convergence_values,
          {five_row_label : average_iter_five_row_mat, seven_row_label : average_iter_seven_row_mat},
          "Convergence Threshold", "Average Number of Iterations",
          "Average Iterations to Convergence By Convergence Value", True)

    five_row_average_minimum_correlations = []
    five_row_average_average_correlations = []
    seven_row_average_minimum_correlations = []
    seven_row_average_average_correlations = []

    for convergence_value in convergence_values:
        five_row_average_minimum, five_row_average_average = getAverageMinimumAndAverageAverageCorrelationValueForResults(five_row_results_by_convergence_value[convergence_value])
        five_row_average_minimum_correlations.append(five_row_average_minimum)
        five_row_average_average_correlations.append(five_row_average_average)

        seven_row_average_minimum, seven_row_average_average = getAverageMinimumAndAverageAverageCorrelationValueForResults(seven_row_results_by_convergence_value[convergence_value])
        seven_row_average_minimum_correlations.append(seven_row_average_minimum)
        seven_row_average_average_correlations.append(seven_row_average_average)

    plotGraph(convergence_values,
          {five_row_label : five_row_average_minimum_correlations, seven_row_label : seven_row_average_minimum_correlations},
          "Convergence Threshold", "Minimum Correlation (Average over 10 sets)",
          "Minimum Correlation (averaged over 10 sets) By Convergence Value", True)

    plotGraph(convergence_values,
          {five_row_label : five_row_average_average_correlations, seven_row_label : seven_row_average_average_correlations},
          "Convergence Threshold", "Average Correlation (Average over 10 sets)",
          "Average Correlation (averaged over 10 sets) By Convergence Value", True)

def runConvergenceThresholdExperiments(sound_data_mat, display_plots):
    """
    Run the convergence threshold experiments. Tries a number of convergence thresholds for 5 and 7 mixed signal
    matrices and runs ICA. Optionally displays plots for the results.

    Args:
        sound_data_mat (2D Numpy Array):    Original signals to mix and then recover using ICA.
        display_plots (boolean):            True if the results should be plotted, false otherwise.
    """
    convergence_values = [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    num_seven_row_mats = 10
    num_five_row_mats = 10
    learning_rate = 0.00001
    max_iterations = 1000000

    # NOTE: These values are here only for submission and fast execution. Results in report use values above.
    # These should be removed to replicate the experiments
    convergence_values = [0.0000001, 0.000001]
    num_seven_row_mats = 2
    num_five_row_mats = 2

    seven_row_mixing_mats = [createRandomSoundMixingMatrixOfSize((7, 5))for i in range(num_seven_row_mats)]
    five_row_mixing_mats = [createRandomSoundMixingMatrixOfSize((5, 5)) for i in range(num_five_row_mats)]

    five_row_results_by_convergence_value = {}
    seven_row_results_by_convergence_value = {}

    for convergence_threshold in convergence_values:
        print("Convergence Threshold: " + str(convergence_threshold))
        five_row_results_by_convergence_value[convergence_threshold] = []
        seven_row_results_by_convergence_value[convergence_threshold] = []
        for seven_row_mix in seven_row_mixing_mats:
            run_config = ICARunConfiguration(seven_row_mix, sound_data_mat, max_iterations,
                                             convergence_threshold, learning_rate)
            seven_row_results_by_convergence_value[convergence_threshold].append(resultsForSingleRun(run_config))

        for five_row_mix in five_row_mixing_mats:
            run_config = ICARunConfiguration(five_row_mix, sound_data_mat, max_iterations,
                                             convergence_threshold, learning_rate)
            five_row_results_by_convergence_value[convergence_threshold].append(resultsForSingleRun(run_config))
            print(five_row_results_by_convergence_value[convergence_threshold][len(five_row_results_by_convergence_value[convergence_threshold]) - 1].ica_results.correlation_values)

    # Uncomment to write to file
    # new_fpath = "assign_2_results_by_convergence_threshold" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".pkl"
    # joblib.dump((convergence_values, five_row_results_by_convergence_value, seven_row_results_by_convergence_value), new_fpath)

    if (display_plots):
        displayConvergenceThresholdExperimentData(convergence_values, five_row_results_by_convergence_value, seven_row_results_by_convergence_value)


def displayLearningRateExperiementData(learning_rates, five_row_results_by_learning_rate, seven_row_results_by_learning_rate):
    """
    Display learning rate experiment result data:

    Args:
        learning_rates (1D Numpy Array):    Learning rates to display results for.
        five_row_results_by_learning_rate (dict with double keys and ICARunData lists as values):   Results for
            mixing matrices that generate 5 mixed signals, stored by the learning rate for which they were
            executed.
        seven_row_results_by_learning_rate (dict with double keys and ICARunData lists as values):  Results for
            mixing matrices that generate 7 mixed signals, stored by the learning rate for which they were
            executed.
    """

    # Get average iterations for each convergence value for 5 row mixing mats and 7 row mixing mats
    average_iter_five_row_mat = [getAverageIterationsForResults(five_row_results_by_learning_rate[learning_rate]) for learning_rate in learning_rates]
    average_iter_seven_row_mat = [getAverageIterationsForResults(seven_row_results_by_learning_rate[learning_rate]) for learning_rate in learning_rates]

    five_row_label = "5 mixed signals"
    seven_row_label = "7 mixed signals"

    plotGraph(learning_rates,
              {five_row_label : average_iter_five_row_mat, seven_row_label : average_iter_seven_row_mat},
              "Learning Rate", "Average Number of Iterations",
              "Average Iterations to Convergence By Learning Rate", True)


    five_row_average_minimum_correlations = []
    five_row_average_average_correlations = []
    seven_row_average_minimum_correlations = []
    seven_row_average_average_correlations = []

    for learning_rate in learning_rates:
        five_row_average_minimum, five_row_average_average = getAverageMinimumAndAverageAverageCorrelationValueForResults(five_row_results_by_learning_rate[learning_rate])
        five_row_average_minimum_correlations.append(five_row_average_minimum)
        five_row_average_average_correlations.append(five_row_average_average)

        seven_row_average_minimum, seven_row_average_average = getAverageMinimumAndAverageAverageCorrelationValueForResults(seven_row_results_by_learning_rate[learning_rate])
        seven_row_average_minimum_correlations.append(seven_row_average_minimum)
        seven_row_average_average_correlations.append(seven_row_average_average)

    plotGraph(learning_rates,
          {five_row_label : five_row_average_minimum_correlations, seven_row_label : seven_row_average_minimum_correlations},
          "Learning Rate", "Minimum Correlation (Average over 10 sets)",
          "Minimum Correlation (averaged over 10 sets) By Learning Rate", True)

    plotGraph(learning_rates,
          {five_row_label : five_row_average_average_correlations, seven_row_label : seven_row_average_average_correlations},
          "Learning Rate", "Average Correlation (Average over 10 sets)",
          "Average Correlation (averaged over 10 sets) By Learning Rate", True)

def displayNumMixedSignalExperimentData(mix_counts, results_by_mix_count):
    """
    Display experiment result data for experiments that varied the number of mixed signals used to recover the original signals.

    Args:
        mix_counts (1D Numpy Array):    Number of mixed signals used in ICA that results should be displayed for.
        results_by_mix_count (dict with double keys and ICARunData lists as values):   Results for
            ICA executions, stored by the number of mixed signals used in the execution.
    """

    average_iter = [getAverageIterationsForResults(results_by_mix_count[mix_count]) for
                                 mix_count in mix_counts]

    plotGraph(mix_counts, {"Average iterations":average_iter}, "Number of Mixed Signals (rows of matrix X)",
              "Average Number of Iterations", "Average Iterations to Convergence By Number of Mixed Signals")

    average_minimum_correlations = []
    average_correlations = []

    for mix_count in mix_counts:
        average_minimum, average_average = getAverageMinimumAndAverageAverageCorrelationValueForResults(
            results_by_mix_count[mix_count])
        average_minimum_correlations.append(average_minimum)
        average_correlations.append(average_average)

    minimum_label = "Minimum Correlation"
    average_label = "Average Correlation"
    plotGraph(mix_counts, {minimum_label: average_minimum_correlations, average_label: average_correlations},
              "Learning Rate", "Correlation (Average over 10 sets)",
              "Correlation (averaged over 10 sets) By Number of Mixed Signals")

def runLearningRateExperiments(sound_data_mat, display_plots):
    """
    Run the learning rate experiments. Tries a number of learning rates for 5 and 7 mixed signal
    matrices and runs ICA. Optionally displays plots for the results.

    Args:
        sound_data_mat (2D Numpy Array):    Original signals to mix and then recover using ICA.
        display_plots (boolean):            True if the results should be plotted, false otherwise.
    """

    print("Testing with varying learning rates")
    convergence_threshold = 0.00000001
    max_iterations = 100000

    learning_rates = [0.000000001, 0.000000005, 0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001]

    num_seven_row_mats = 5
    num_five_row_mats = 5

    # NOTE: These values are here only for submission and fast execution. Results in report use values above.
    # These should be removed to replicate the experiments
    learning_rates = [0.000001, 0.00001]
    num_seven_row_mats = 2
    num_five_row_mats = 2

    seven_row_mixing_mats = [createRandomSoundMixingMatrixOfSize((7, 5)) for i in range(num_seven_row_mats)]
    five_row_mixing_mats = [createRandomSoundMixingMatrixOfSize((5, 5)) for i in range(num_five_row_mats)]

    five_row_results_by_learning_rate = {}
    seven_row_results_by_learning_rate = {}

    for learning_rate in learning_rates:
        print("Learning rate: " + str(learning_rate))
        five_row_results_by_learning_rate[learning_rate] = []
        seven_row_results_by_learning_rate[learning_rate] = []
        for seven_row_mix in seven_row_mixing_mats:
            run_config = ICARunConfiguration(seven_row_mix, sound_data_mat, max_iterations,
                                             convergence_threshold, learning_rate)
            seven_row_results_by_learning_rate[learning_rate].append(resultsForSingleRun(run_config))

        for five_row_mix in five_row_mixing_mats:
            run_config = ICARunConfiguration(five_row_mix, sound_data_mat, max_iterations,
                                             convergence_threshold, learning_rate)
            five_row_results_by_learning_rate[learning_rate].append(resultsForSingleRun(run_config))

    # Uncomment to write data to file
    # new_fpath = "assign_2_results_by_num_learning_rates" + datetime.datetime.now().replace(
    #     microsecond=0).isoformat() + ".pkl"
    # joblib.dump((learning_rates, five_row_results_by_learning_rate, seven_row_results_by_learning_rate), new_fpath)

    if (display_plots):
        displayLearningRateExperiementData(learning_rates, five_row_results_by_learning_rate, seven_row_results_by_learning_rate)

def runNumMixedSignalExperiments(sound_data_mat, display_plots):
    """
    Run the number of mixed signals experiments. Tries different numbers of mixed signals and runs ICA on the mixed
    signal data. Optionally displays plots for the results.

    Args:
        sound_data_mat (2D Numpy Array):    Original signals to mix and then recover using ICA.
        display_plots (boolean):            True if the results should be plotted, false otherwise.
    """

    num_matrices_for_each_mix_count = 10
    mix_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

    # NOTE: These values are here only for submission and fast execution. Results in report use values above.
    # These should be removed to replicate the experiments
    mix_counts = [4, 5, 6]
    num_matrices_for_each_mix_count = 3

    mixing_matrices_by_mix_count = {}
    for mix_count in mix_counts:
        mixing_matrices_by_mix_count[mix_count] = [createRandomSoundMixingMatrixOfSize((mix_count, 5)) for i in range(num_matrices_for_each_mix_count)]

    convergence_threshold = 0.00000001
    max_iterations = 100000
    learning_rate = 0.00001

    results_by_mix_count = {}
    for mix_count in mix_counts:
        print("Mix Count: " + str(mix_count))
        results_by_mix_count[mix_count] = []
        for mixing_mat in mixing_matrices_by_mix_count[mix_count]:
            run_config = ICARunConfiguration(mixing_mat, sound_data_mat, max_iterations, convergence_threshold,
                                             learning_rate)
            results_by_mix_count[mix_count].append(resultsForSingleRun(run_config))

    # Uncomment to write data to file
    # new_fpath = "assign_2_results_by_num_mixed_signals" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".pkl"
    # joblib.dump((mix_counts, results_by_mix_count), new_fpath)

    if (display_plots):
        displayNumMixedSignalExperimentData(mix_counts, results_by_mix_count)


# def plotData():
#     """
#     Plot data from previously saved data. Not used in primary assignment execution.
#     """
#
#     learning_rate_file = "assign_2_results_by_num_learning_rates2020-09-24T20:01:49.pkl"
#     # o_g_learning_rates = [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
#     # learning_rates = [0.000000001, 0.000000005, 0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001]
#     # learning_rates = [0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001]
#     (learning_rates, five_row_results_by_learning_rate, seven_row_results_by_learning_rate) = joblib.load(learning_rate_file)
#
#     # displayLearningRateExperiementData(learning_rates, five_row_results_by_learning_rate, seven_row_results_by_learning_rate)
#
#     # convergence_file = "assign_2_results_by_convergence_threshold2020-09-23T15:20:31.pkl"
#     # convergence_values = [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01,
#     # 0.1] # TODO is this good?
#     # convergence_values = [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
#     convergence_file = "assign_2_results_by_convergence_threshold2020-09-24T22:01:31.pkl"
#     (convergence_values, five_row_results_by_convergence_value, seven_row_results_by_convergence_value) = joblib.load(convergence_file)
#     displayConvergenceThresholdExperimentData(convergence_values, five_row_results_by_convergence_value, seven_row_results_by_convergence_value)
#
#     mix_file = "assign_2_results_by_num_mixed_signals2020-09-23T16:06:31.pkl"
#     # mix_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
#     # mix_counts = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
#     (mix_counts, results_by_mix_count) = joblib.load(mix_file)
#     displayNumMixedSignalExperimentData(mix_counts, results_by_mix_count)


def executeAssignment2(large_data_set_file_name, small_dataset_file_name, display_plots=False):

    """
    Run all experiments and plots the signals in both the small and large data sets.

    Args:
        large_data_set_file_name (string):  Name of the file containing the large data set.
        small_dataset_file_name (string):   Name of the file containing the small data set.
        display_plots (boolean):            True if the plots should be displayed, false if the plots should be skipped.
    """

    # Read the large dataset
    sound_data_mat = readSoundData(large_data_set_file_name)
    if (display_plots):
        plotSmallSetAndLargeSetWithDifferentAValues(sound_data_mat, small_dataset_file_name)

    runConvergenceThresholdExperiments(sound_data_mat, display_plots)
    runNumMixedSignalExperiments(sound_data_mat, display_plots)
    runLearningRateExperiments(sound_data_mat, display_plots)


if __name__=="__main__":

    parser_results = getCmdLineArgs()

    sound_file_name = parser_results.sound_file_name
    small_dataset_file_name = parser_results.small_dataset_file_name

    executeAssignment2(sound_file_name, small_dataset_file_name, True)



