
import numpy as np
from scipy.special import expit as sigmoid
import argparse
import scipy.io
import matplotlib.pyplot as plt
import scipy.signal

def readSoundData(sound_file_name):
    """
    Read the data from the given file name. Assuming it is a MATLAB file.

    :param sound_file_name: Name of the file containing the sound data.

    :return: Numpy matrix containing the data, with each row of the matrix being one audio signal
    """

    data_in_file = scipy.io.loadmat(sound_file_name)
    sounds_mat_key = "sounds"
    sounds_mat = data_in_file[sounds_mat_key]
    print("Sounds mat shape")
    print(sounds_mat.shape)
    # return normalizeSoundMatrixRows(sounds_mat) # TODO investigate if this should be normalized... seemed to do bad things in test mode
    return sounds_mat

def readTestData(test_data_file_name):
    data_in_file = scipy.io.loadmat(test_data_file_name)
    a_mat_key = "A"
    u_mat_key = "U"
    return (data_in_file[a_mat_key], data_in_file[u_mat_key])

def createSoundMixingMatricesToEvaluate(mat_size = None):

    # TODO add more matrices?

    if mat_size is None:
        mat_size = (7, 5)

    return [np.random.uniform(0, 1, mat_size)]


def plotSignals(original_signals, mixed_signals, extracted_signals):
    num_original_signals = original_signals.shape[0]
    num_mixed_signals = mixed_signals.shape[0]
    num_extracted_signals = extracted_signals.shape[0]
    signal_len = original_signals.shape[1]
    normalized_mixed = normalizeSoundMatrixRows(mixed_signals)
    normalized_original = normalizeSoundMatrixRows(original_signals)

    fig, ax = plt.subplots(1)
    num_plotted_signals = 0
    for i in range(num_original_signals):
        plt.plot(normalized_original[i] + num_plotted_signals)
        num_plotted_signals += 1
    for i in range(num_extracted_signals):
        plt.plot(extracted_signals[i] + num_plotted_signals)
        num_plotted_signals += 1
    for i in range(num_mixed_signals):
        plt.plot(normalized_mixed[i] + num_plotted_signals)
        num_plotted_signals += 1
    left, width = -8, .5
    bottom, height = 0, (num_extracted_signals + num_mixed_signals + num_original_signals)
    right = left + width
    top = bottom + height


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
    plt.hlines([num_original_signals, (num_extracted_signals + num_original_signals)], xmin=0, xmax=signal_len)
    plt.show()

def initializeUnmixingMatrix(num_rows, num_columns):
    return np.random.uniform(0.0, 0.1, (num_rows, num_columns))

def calculateZMat(unmixed_estimate):
    """
    TODO
    :param unmixed_estimate:
    :return:
    """
    return sigmoid(unmixed_estimate)

def calculateGradientW(learning_rate, unmixed_estimate_y, z_mat, unmixing_matrix_w):

    # TODO verify this
    # identity times t
    multiplied_identity = np.identity(unmixing_matrix_w.shape[0]) * unmixed_estimate_y.shape[1]

    grad_w = learning_rate * (multiplied_identity + ((1 - (2 * z_mat)) @ np.transpose(unmixed_estimate_y))) @ unmixing_matrix_w
    return grad_w

def checkForConvergence(updated_unmixed_estimate, prev_unmixed_estimate):
    convergence_threshold = 0.000001 # TODO is this low enough?
    abs_mat_diff = abs(updated_unmixed_estimate - prev_unmixed_estimate)
    print("Convergence check")
    print(abs_mat_diff)
    # print("Convergence?")
    # print((abs_mat_diff < convergence_threshold).all())
    # if ((abs_mat_diff < convergence_threshold).all()):
    #     print(abs_mat_diff)
    #
    # # TODO
    return (abs_mat_diff < convergence_threshold).all()

def normalizeSoundMatrixRows(sound_matrix):
    # print("Normalizing")
    # print(sound_matrix)
    # print("Max/min per row")
    max_per_row = np.amax(sound_matrix, axis=1, keepdims=True)
    min_per_row = np.amin(sound_matrix, axis=1, keepdims=True)
    # print(max_per_row)
    # print(min_per_row)

    range_per_row = max_per_row - min_per_row
    # print(range_per_row)
    min_subtracted_mat = sound_matrix - min_per_row
    # print("min subtracted")
    # print(min_subtracted_mat)
    normalized_mat = min_subtracted_mat / range_per_row
    # print("Done normalizing")
    return normalized_mat


def extractOriginalSounds(mixed_sounds, learning_rate, max_iterations, num_original_sounds):

    # Step 2 of Gradient Descent Method
    unmixing_matrix_w = initializeUnmixingMatrix(num_original_sounds, mixed_sounds.shape[0])
    m_num_mixed_sounds = mixed_sounds.shape[0]
    sound_length = mixed_sounds.shape[1]
    print(unmixing_matrix_w)

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
        updated_unmixing_matrix = unmixing_matrix_w + gradient_w

        # check for convergence
        converged = checkForConvergence(updated_unmixing_matrix, unmixing_matrix_w)
        unmixing_matrix_w = updated_unmixing_matrix
        iterations += 1
        print("Iterations " + str(iterations))

    return normalizeSoundMatrixRows(unmixing_matrix_w @ mixed_sounds), unmixing_matrix_w

def writeSignalsToFile(signal_data):
    pass




def mixAndUnmixAndPlotResultsForLearningRates(learning_rates, mixing_mat, original_sounds, max_iterations):
    print(mixing_mat)
    # Mix to get new matrix X
    # mixed_data = normalizeSoundMatrixRows(mixing_mat @ original_sounds) # This produces bad results for some reason in testing mode...?
    mixed_data = mixing_mat @ original_sounds
    num_original_sounds = original_sounds.shape[0]
    sound_length = original_sounds.shape[1]
    # Play/display mixed vs original?
    # Try to unmix
    for learning_rate in learning_rates:
        unmixed_data_estimate = extractOriginalSounds(mixed_data, learning_rate, max_iterations, num_original_sounds)
        unmixed_sounds = unmixed_data_estimate[0]
        print("Unmixed sounds")
        print(unmixed_sounds.shape)
        plotSignals(original_sounds, mixed_data, unmixed_sounds)
        correlation = scipy.signal.correlate2d(original_sounds,unmixed_sounds)
        print("Correlation: ")
        print(correlation)

        # Play/display original vs unmixed?

def executeAssignmentTwoFullVersion(sound_file_name):

    sound_data_mat = readSoundData(sound_file_name)
    sound_mixing_matrices = createSoundMixingMatricesToEvaluate()

    # TODO how to determine learning rate? Should this try multiple learning rates?
    learning_rates = [0.000001]

    # TODO set max iterations more logically
    max_iterations = 10000

    for sound_mixing_mat in sound_mixing_matrices:
        mixAndUnmixAndPlotResultsForLearningRates(learning_rates, sound_mixing_mat, sound_data_mat, max_iterations)


def executeTestMatVersion():
    test_mat_file_name = "/Users/mandiadkins/Downloads/icaTest.mat"

    mixing_matrix, original_sound_matrix = readTestData(test_mat_file_name)
    max_iterations = 10000
    learning_rates = [0.01]

    mixAndUnmixAndPlotResultsForLearningRates(learning_rates, mixing_matrix, original_sound_matrix, max_iterations)


def getCmdLineArgs():
    """
    TODO
    :return:
    """

    default_sound_file_name = "/Users/mandiadkins/Downloads/sounds.mat"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--sound_file_name", default=default_sound_file_name, help='Name of the file '\
        'containing the sound clips to mix and unmix.')
    arg_parser.add_argument("--use_test_mat", action='store_true', help='Use the test data (small data set) instead')

    return arg_parser.parse_args()

if __name__=="__main__":

    parser_results = getCmdLineArgs()

    sound_file_name = parser_results.sound_file_name
    use_test_mat = parser_results.use_test_mat
    # use_test_mat = True

    if (use_test_mat):
        executeTestMatVersion()
    else:
        executeAssignmentTwoFullVersion(sound_file_name)
    #
    # test_mat = np.random.randint(0, 10, size=(3, 5))
    # print(normalizeSoundMatrixRows(test_mat))


