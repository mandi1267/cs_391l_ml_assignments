"""
Amanda Adkins
Fall 2020 CS391L Assignment 4
"""

import argparse
import csv
from gaussian_process_predictor import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

import math
import datetime

import joblib

class SingleCoordInfo:
    """
    Contains the data for a single coordinate.
    """
    def __init__(self, elapsed_time, coordinate_val, is_valid):
        self.elapsed_time = elapsed_time
        self.coordinate_val = coordinate_val
        self.is_valid = is_valid

class Pose:
    def __init__(self, x_pos, y_pos, z_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.z_pos = z_pos

    def __str__(self):
        return "Pose(" + str(self.x_pos) + ", " + str(self.y_pos) + ", " + str(self.z_pos) + ")"

class MarkerPos:
    def __init__(self, marker_num, marker_pose, is_valid):
        self.marker_num = marker_num
        self.marker_pose = marker_pose
        self.is_valid = is_valid

    def __str__(self):
        return "MarkerPos(num: " + str(self.marker_num) + ", pos: " + str(self.marker_pose) + ", valid: " + str(self.is_valid) + ")"

class MarkersAtTime:
    def __init__(self, frame, elapsed_time, target_pose, finger_pose, head_pose, marker_poses_by_marker_num):
        self.frame = frame
        self.elapsed_time = elapsed_time
        self.target_pose = target_pose
        self.finger_pose = finger_pose
        self.head_pose = head_pose
        self.marker_poses_by_marker_num = marker_poses_by_marker_num

    def __str__(self):
        return "MarkersAtTime(frame: " + str(self.frame) + ", el_time: " + str(self.elapsed_time) + ", target: " + str(self.target_pose) + ", finger: " + str(self.finger_pose) + ", head: " + str(self.head_pose) + ")"

class MarkerTraces:
    def __init__(self, markers_at_each_time):
        self.markers_at_each_time = markers_at_each_time

    def getPoseSequenceForMarkerNum(self, marker_num):
        return [(timestep.elapsed_time, timestep.marker_poses_by_marker_num[marker_num]) for timestep in self.markers_at_each_time]

    def getCoordSequenceForMarkerNumAndCoordinate(self, marker_num, coord_string):
        pose_seq = self.getPoseSequenceForMarkerNum(marker_num)
        if (coord_string == "x"):
            return [SingleCoordInfo(timestamp, pose_info.marker_pose.x_pos, pose_info.is_valid) for (timestamp, pose_info) in pose_seq]
        elif (coord_string == "y"):
            return [SingleCoordInfo(timestamp, pose_info.marker_pose.y_pos, pose_info.is_valid) for (timestamp, pose_info) in pose_seq]
        elif (coord_string == "z"):
            return [SingleCoordInfo(timestamp, pose_info.marker_pose.z_pos, pose_info.is_valid) for (timestamp, pose_info) in pose_seq]
        else:
            return []

    def getNumTimesteps(self):
        return len(self.markers_at_each_time)

def readMarkersAtTime(comma_separated_line):
    """
    Read the marker positions from one line of the CSV.

    Args:
        comma_separated_line (list of strings): List of strings from one line of the file.

    Returns:
        Marker positions for one timestamp (contained in a MarkersAtTime object).
    """
    i = 0
    frame_num = int(comma_separated_line[i])
    i+= 1
    elapsed_time = float(comma_separated_line[i])
    i += 1
    target_pose = Pose(float(comma_separated_line[i]), float(comma_separated_line[i+1]), float(comma_separated_line[i+2]))
    i += 3
    finger_pose = Pose(float(comma_separated_line[i]), float(comma_separated_line[i+1]), float(comma_separated_line[i+2]))
    i += 3
    head_pose = Pose(float(comma_separated_line[i]), float(comma_separated_line[i + 1]), float(comma_separated_line[i + 2]))
    i += 3

    num_markers = 50

    marker_poses_by_marker_num = {}

    for marker_num in range(num_markers):
        line_index = i + 4 * marker_num
        marker_pos = Pose(float(comma_separated_line[line_index]), float(comma_separated_line[line_index + 1]), float(comma_separated_line[line_index + 2]))
        is_valid = (float(comma_separated_line[line_index + 3]) > 0)
        marker_info = MarkerPos(marker_num, marker_pos, is_valid)
        marker_poses_by_marker_num[marker_num] = marker_info

    markers_at_time = MarkersAtTime(frame_num, elapsed_time, target_pose, finger_pose, head_pose, marker_poses_by_marker_num)

    return markers_at_time

def readData(file_name):
    """
    Read marker data from the given filename.

    Args:
        file_name (string): Name of the file to read marker data from.
    Returns:
        Marker traces object containing the marker data from the file.
    """
    marker_traces_list = []
    with open(file_name) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        first_line = True
        for csv_line in csv_reader:
            if (first_line):
                first_line = False
            else:
                marker_traces_list.append(readMarkersAtTime(csv_line))
    return MarkerTraces(marker_traces_list)

def plotTrain(timesteps, values):
    plt.plot(timesteps, values, 'bo')
    plt.show()


def plotPredictionVsActual(timesteps, predicted_values, predicted_std_dev, actual_values, marker_num, coord_string,
                           train_timestamps, train_values):

    """
    Plot the training data, actual test data, predicted values for the test data, and standard deviations for the test
    data.

    Args:
        timestamps (2D Numpy array):            Timestamps for the test data
        predicted_values (1D Numpy array):      Predicted values for the test data.
        predicted_std_dev (1D Numpy array):     Standard deviation output by the regressor for the predicted test data.
        actual_values (2D Numpy array):         Actual output for the test data. Should have one column.
        marker_num (int):                       Marker number that the data is for.
        coord_string (string):                  String (x, y, or z) marking the coordinate that the data is for.
        train_timestamps (2D Numpy array):      Training data timestamps.
        train_values (2D Numpy array):          Output value for the training data.
    """

    plt.figure()

    plt.plot(train_timestamps, train_values, 'g,', label='Training')
    plt.plot(timesteps, predicted_values, 'b-', label='Prediction')
    plt.plot(timesteps, actual_values, 'r-', label='Actual')
    plt.fill(np.concatenate([timesteps, timesteps[::-1]]),
             np.concatenate([predicted_values - predicted_std_dev,
                             (predicted_values + predicted_std_dev)[::-1]]),
             alpha=.5, fc='b', ec='None', label='+/- 1 std dev')
    plt.xlabel('Elapsed Time')
    plt.ylabel(coord_string + ' coord for marker ' + str(marker_num))
    plt.legend(loc='upper left')
    plt.title("Predicted vs Actual position for " + coord_string + " coord of marker " + str(marker_num) + " with training data")
    plt.show()

    plt.plot(timesteps, predicted_values, 'b-', label='Prediction')
    plt.plot(timesteps, actual_values, 'r-', label='Actual')
    plt.fill(np.concatenate([timesteps, timesteps[::-1]]),
             np.concatenate([predicted_values - predicted_std_dev,
                             (predicted_values + predicted_std_dev)[::-1]]),
             alpha=.5, fc='b', ec='None', label='+/- 1 std dev')
    plt.xlabel('Elapsed Time')
    plt.ylabel(coord_string + ' coord for marker ' + str(marker_num))
    plt.legend(loc='upper left')
    plt.title("Predicted vs Actual position for " + coord_string + " coord of marker " + str(marker_num))
    plt.show()


def computeTrajectoryError(actual_trajectory, predicted_trajectory):

    """
    Compute the mean squared error for the test trajectory prediction.

    Args:
        actual_trajectory (2D Numpy array):    Actual data for the test trajectory.
        predicted_trajectory (2D Numpy array): Predicted data for the test trajectory.

    Returns:
        Mean squared error for the trajectory prediction.
    """
    # Assuming trajectories only have one coordinate (x, y, or z, not all 3)
    squared_error_dist_sum = 0
    for i in range(len(predicted_trajectory)):
        act_pose = actual_trajectory[i]
        pred_pose = predicted_trajectory[i]
        squared_error = pow(act_pose - pred_pose, 2)
        squared_error_dist_sum += squared_error
    mean_squared_error = squared_error_dist_sum / len(predicted_trajectory)

    return mean_squared_error

def plotMarkerPos(pose_seq, marker_num, data_set_name):
    """
    Plot the 3D marker trace.

    Args:
        pose_seq (list of tuples):  3D Pose sequence to plot. Each entry in the list is a tuple of the time for the
                                    pose and the 3D pose data.
        marker_num (int):           Marker number (used for display)
        data_set_name (string):     Name of the data set to display when plotting
    """

    valid_x = []
    valid_y = []
    valid_z = []
    for (timestamp, pose_data) in pose_seq:
        if (pose_data.is_valid):
            pose = pose_data.marker_pose
            valid_x.append(pose.x_pos)
            valid_y.append(pose.y_pos)
            valid_z.append(pose.z_pos)

    title_str = "Pose sequence for marker " + str(marker_num) + " in dataset\n" + data_set_name
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot(valid_x, valid_y, valid_z, 'blue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title_str)
    plt.show()

def convertMarkerPosSeqToGPForamt(marker_data_sequences, marker_num, coord_string):
    """
    Convert marker data to the format needed by the Gaussian process regressor (also filter out invalid coordinates).

    Args:
        marker_data_sequences (List of MarkerTraces objects):   Contains the data sequences to convert.
        marker_num (int):                                       Marker number to extract.
        coord_string (string):                                  Coordinate string indicating which coordinate to
                                                                extract. Must be "x", "y", or "z"
    Returns:
        Tuple of numpy arrays containing the x data (timestamp) and y data (coordinate value) for the marker data
        sequences.
    """
    x_data = []
    y_data = []
    for marker_data in marker_data_sequences:
        coord_seq = marker_data.getCoordSequenceForMarkerNumAndCoordinate(marker_num, coord_string)
        for coord in coord_seq:
            if (coord.is_valid):
                y_data.append(coord.coordinate_val)
                x_data.append(coord.elapsed_time)
            else:
                print("Ignoring invalid coord at time " + str(coord.elapsed_time))

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data = x_data.reshape(-1, 1)
    y_data = y_data.reshape(-1, 1)
    return (x_data, y_data)

def getDataInTimeRange(times, values, min_time, max_time):
    """
    Gets the data for timestamps in the supplied range. Gets the timestamps and values corresponding to the matching
    timestamps.

    Args:
        times (2D Numpy array with only 1 column):  Times for the data
        values (2D Numpy array with only 1 column): Value for the data
        min_time (float):                           Minimum time to include in the returned data.
        max_time (float):                           Maximum time to include in the returned data.
    Returns:
        Tuple of the times in the range and the values in the range (both are 2D numpy arrays iwth only 1 column).
    """
    # Assumes that values and times are sorted by time
    times_in_range = []
    values_in_range = []
    for i in range(len(times)):
        if (times[i] > max_time):
            break
        if (times[i] >= min_time):
            times_in_range.append(times[i])
            values_in_range.append(values[i])

    times_in_range = np.vstack(times_in_range)
    values_in_range = np.vstack(values_in_range)
    times_in_range = times_in_range.reshape(-1, 1)
    values_in_range = values_in_range.reshape(-1, 1)
    return times_in_range, values_in_range

def plotHyperparamsBySlidingWindow(hyperparam_results_for_sliding_window):
    """
    Plot the hyperparameters for the sliding windows.

    Args:
        hyperparam_results_for_sliding_windows (list of 3 element tuples): Hyperparameter results and the sliding
            windows for which they apply to. Each tuple in the list contains the start time for the window, the end time
            for the window, and the hyperparameters used in that window. The windows may or may not overlap.
    """
    rbf_len_lines = []
    rbf_var_lines = []
    noise_lines = []
    colors = []

    for window_start, window_end, hyperparam_results in hyperparam_results_for_sliding_window:
        rbf_len = hyperparam_results.rbf_len
        rbf_var = hyperparam_results.rbf_variance
        noise = hyperparam_results.noise

        # TODO Param boundaries may need to be modified
        if (rbf_len < 1000):
            rbf_len_lines.append([(window_start, hyperparam_results.rbf_len), (window_end, hyperparam_results.rbf_len)])
        if (rbf_var < 1000):
            rbf_var_lines.append([(window_start, hyperparam_results.rbf_variance), (window_end, hyperparam_results.rbf_variance)])
        if (noise < 1000):
            noise_lines.append([(window_start, hyperparam_results.noise), (window_end, hyperparam_results.noise)])
        colors.append((1, 0, 0, 1))


    lc = mc.LineCollection(rbf_len_lines, colors=colors, linewidths=2)
    # print(rbf_len_lines)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.set_title("RBF Len")
    ax.autoscale()
    plt.show()

    lc = mc.LineCollection(rbf_var_lines, colors=colors, linewidths=2)
    # print(rbf_var_lines)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.set_title("RBF Var")
    ax.autoscale()
    plt.show()

    lc = mc.LineCollection(noise_lines, colors=colors, linewidths=2)
    fig, ax = plt.subplots()
    # print(noise_lines)
    ax.add_collection(lc)
    ax.set_title("Noise")
    ax.autoscale()
    plt.show()


def chooseWindows(hyperparam_results, new_window_threshold, train_timestamps, train_values):
    """
    Choose the window boundaries given the hyperparameter results from the more granular sliding window sweep.

    Args:
        hyperparam_results:     Tuple of window start, window end, and hyperparameters for the window
        new_window_threshold:   Difference between merged window average RBF length and new window RBF length that
                                triggers a window boundary.
        train_timestamps:       Timestamps of the training data.
        train_values:           Values for the training data.

    Returns:
        Trained model with submodels for non-overlapping time windows.
    """
    window_starts = [0.0]
    curr_window_rbf_sum = 0.0
    curr_window_rbf_count = 0.0

    prev_window_end = 0

    for window_start, window_end, hyperparams in hyperparam_results:
        if (window_start < window_starts[-1]):
            continue
        if (curr_window_rbf_count == 0):
            curr_window_rbf_sum = hyperparams.rbf_len
            curr_window_rbf_count += 1
            prev_window_end = window_end
        else:
            if (abs(hyperparams.rbf_len - (curr_window_rbf_sum / curr_window_rbf_count)) > new_window_threshold):
                curr_window_rbf_count = 0
                window_starts.append(prev_window_end)
            else:
                curr_window_rbf_count += 1
                curr_window_rbf_sum += hyperparams.rbf_len
                prev_window_end = window_end

    # Retrain using window start to next window start
    updated_models = []
    plottable_hyperparam_results = []
    default_init_hyperparams = GaussianProcessHyperparams(1.0, 0.05, 0.01)
    print("Number of different windows")
    print(len(window_starts))
    for i in range(len(window_starts)):
        if (i == 0):
            window_start = 0.0
        else:
            window_start = window_starts[i]
        if (i == (len(window_starts) - 1)):
            window_end = train_timestamps[-1]
        else:
            window_end = window_starts[i + 1]
        times_in_range, values_in_range = getDataInTimeRange(train_timestamps, train_values, window_start, window_end)
        print("Retraining: Window start, window end: " + str(window_start) + ", " + str(window_end))

        sliding_window_gp = GaussianProcessModel(default_init_hyperparams, 5, True)
        sliding_window_gp.trainModel(times_in_range, values_in_range)
        updated_models.append((window_start, sliding_window_gp))
        plottable_hyperparam_results.append((window_start, window_end, sliding_window_gp.getHyperParams()))
    # plotHyperparamsBySlidingWindow(plottable_hyperparam_results)

    return GaussianProcessModelWithVaryingHyperparameters(updated_models)


def trainWithSpecifiedWindows(train_timestamps, train_values, window_starts):
    updated_models = []
    plottable_hyperparam_results = []
    default_init_hyperparams = GaussianProcessHyperparams(1.0, 0.05, 0.01)
    print("Number of different windows")
    print(len(window_starts))
    for i in range(len(window_starts)):
        if (i == 0):
            window_start = 0.0
        else:
            window_start = window_starts[i]
        if (i == (len(window_starts) - 1)):
            window_end = train_timestamps[-1]
        else:
            window_end = window_starts[i + 1]
        times_in_range, values_in_range = getDataInTimeRange(train_timestamps, train_values, window_start, window_end)
        print("Retraining: Window start, window end: " + str(window_start) + ", " + str(window_end))

        sliding_window_gp = GaussianProcessModel(default_init_hyperparams, 5, True)
        sliding_window_gp.trainModel(times_in_range, values_in_range)
        updated_models.append((window_start, sliding_window_gp))
        plottable_hyperparam_results.append((window_start, window_end, sliding_window_gp.getHyperParams()))
    # plotHyperparamsBySlidingWindow(plottable_hyperparam_results)

    return GaussianProcessModelWithVaryingHyperparameters(updated_models)


def createSlidingWindowGPWrappers(train_data_single_marker_times, train_data_single_marker_single_coords, marker_num,
                                  coord_string, min_window_inc, window_inc_multipliers, sliding_window_size,
                                  rbf_difference_thresholds):

    """
    Create a sliding window GP wrapper for each of the possible window increments (derived from the multipliers) and
    the RBF difference thresholds used to decide the window boundaries.

    Args:
        train_data_single_marker_times:         Training data timestamps
        train_data_single_marker_single_coords: Training data coordinate values
        marker_num:                             Marker num that the data is for
        coord_string:                           Coordinate that the data is for
        min_window_inc:                         Minimum window increment (use this to train the initial submodels).
        window_inc_multipliers:                 Multipliers for the window increment. Window increments are created by
                                                multiplying each of these values with the minimum window increment.
        sliding_window_size:                    Size of the sliding window (in seconds).
        rbf_difference_thresholds:              Threshold for the difference in RBF parameter for merging sliding
                                                windows.
    Returns:
        List of tuples, where each tuple contains the RBF difference threshold, window increment, and the trained model
        that wraps the submodels.
    """

    # Assumes that train times and values are sorted by time
    min_time = train_data_single_marker_times[0][0]
    max_time = train_data_single_marker_times[-1][0]

    window_start = min_time
    window_end = min(max_time, window_start + sliding_window_size)
    continue_loop = True
    not_last_loop = True
    default_init_hyperparams = GaussianProcessHyperparams(1.0, 0.05, 0.01)
    hyperparam_results = []

    while (continue_loop):
        print("Window start, window end " + str(window_start) + ", " + str(window_end))
        times_in_range, values_in_range = getDataInTimeRange(train_data_single_marker_times,
            train_data_single_marker_single_coords, window_start, window_end)

        sliding_window_gp = GaussianProcessModel(default_init_hyperparams, 5, True)
        sliding_window_gp.trainModel(times_in_range, values_in_range)

        hyperparam_results.append((window_start, window_end, sliding_window_gp.getHyperParams()))
        print(str(hyperparam_results[-1][2]))
        continue_loop = not_last_loop
        window_start += min_window_inc
        window_end = window_start + sliding_window_size
        if (window_end > max_time):
            window_end = max_time
            not_last_loop = False


    # Pickle hyperparam results
    results_to_save = {'sliding_window_size': sliding_window_size, 'sliding_window_inc': min_window_inc,
                       'marker_num': marker_num, 'coord_string': coord_string, 'hyperparam_results': hyperparam_results}

    # Uncomment to write to file
    # new_fpath = "sliding_window_results_reusing_inc_" + str(marker_num) + "_" + coord_string + "_" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".pkl"
    # joblib.dump(results_to_save, new_fpath)

    # plotHyperparamsBySlidingWindow(hyperparam_results)

    wrappers = []
    for window_multiplier in window_inc_multipliers:
        reduced_hyperparam_results = []
        window_inc = min_window_inc * window_multiplier

        if (window_inc > sliding_window_size):
            continue

        print("Evaluating for window inc " + str(window_inc))
        for i in range(len(hyperparam_results)):
            if (((i % window_multiplier) == 0) or (i == (len(hyperparam_results) -1))):
                reduced_hyperparam_results.append(hyperparam_results[i])

        for rbf_difference_threshold in rbf_difference_thresholds:
            print("Breaking windows for threshold " + str(rbf_difference_threshold))

            wrappers.append((rbf_difference_threshold, window_inc, chooseWindows(reduced_hyperparam_results, rbf_difference_threshold,
                                                                 train_data_single_marker_times,
                                                                 train_data_single_marker_single_coords)))
    return wrappers


def sortTimestampsAndValuesByTime(timestamps, values):
    """
    Sort the timestamps and values by the timestamps (assuming the value at index i corresponds to the timestamp at
    index i).

    Args:
        timestamps:                 Numpy array containing the timestamps to sort by (should have only one column).
        values (2D Numpy array):    Numpy array containing the values (should have only one column).
    Returns:
        Tuple of the sorted times and sorted values.
    """

    # Sort the training data by the time
    timestamps_and_values = np.hstack((timestamps, values))
    sorted_timestamps_and_values = timestamps_and_values[(timestamps_and_values[:, 0]).argsort()]

    sorted_times = sorted_timestamps_and_values[:, 0]
    sorted_values = sorted_timestamps_and_values[:, 1]

    sorted_times = sorted_times.reshape(-1, 1)
    sorted_values = sorted_values.reshape(-1, 1)

    return (sorted_times, sorted_values)

def executeAssign3WithConfig(train_data_sets, test_data_set, marker_num, coord_string):

    # Get the full 3d pose for the marker for the training and test data
    train_full_3d_pose_for_marker = [data_set.getPoseSequenceForMarkerNum(marker_num) for data_set in train_data_sets]
    test_full_3d_pose_for_marker = test_data_set.getPoseSequenceForMarkerNum(marker_num)

    # plot test 3d pose for marker
    for i in range(len(train_full_3d_pose_for_marker)):
        plotMarkerPos(train_full_3d_pose_for_marker[i], marker_num, "Train sequence " + str(i))
    plotMarkerPos(test_full_3d_pose_for_marker, marker_num, "Test data sequence")

    train_data_single_marker_times, train_data_single_marker_single_coords = convertMarkerPosSeqToGPForamt(train_data_sets, marker_num, coord_string)
    test_data_single_marker_times, test_data_single_marker_single_coords = convertMarkerPosSeqToGPForamt([test_data_set], marker_num, coord_string)

    train_data_single_marker_times, train_data_single_marker_single_coords = sortTimestampsAndValuesByTime(train_data_single_marker_times, train_data_single_marker_single_coords)

    plotTrain(train_data_single_marker_times, train_data_single_marker_single_coords)

    # Train single GP model (same params for whole time)
    gp_initial_hyperparams = GaussianProcessHyperparams(1, 0.5, 0.6)
    global_params_gp_model = GaussianProcessModel(gp_initial_hyperparams, 5, True)
    global_params_gp_model.trainModel(train_data_single_marker_times, train_data_single_marker_single_coords)

    # Get global hyperparams
    gp_model_global_model_wrapper = GaussianProcessModelWithVaryingHyperparameters([(0.0, global_params_gp_model)])
    trajectory_prediction, trajectory_std_dev = gp_model_global_model_wrapper.predictTrajectory(test_data_single_marker_times)

    # Plot prediction + var for test_data times
    plotPredictionVsActual(test_data_single_marker_times, trajectory_prediction, trajectory_std_dev,
                           test_data_single_marker_single_coords, marker_num, coord_string, train_data_single_marker_times, train_data_single_marker_single_coords)

    # Compute MSE for test data coords
    mean_square_error_single_hyperparams = computeTrajectoryError(test_data_single_marker_single_coords, trajectory_prediction)
    print("Mean square error for global set of hyperparams " + str(mean_square_error_single_hyperparams))

    # Note: Use these values instead of those below for a more thorough evaluation
    # window_sizes = [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
    # window_inc_multipliers = [1, 2, 5, 10, 15, 20]
    # min_window_inc = 0.05
    # thresholds = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10]

    min_window_inc = 0.1
    window_sizes = [2]
    window_inc_multipliers = [1]
    thresholds = [0.5]

    mse_by_window_config = []

    best_gp_model_wrapper = None
    best_sliding_gp_model_mse = math.inf

    # Cycle through different combinations of sliding window decision boundary parameters, identify window boundaries
    # based on the hyperparameters for the initial sliding windows, identify new window boundaries and create a model
    # with different submodels for each time window
    # Choose the one that gives the best mean squared error on the test data
    for window_size in window_sizes:
        print("Training for window size: " + str(window_size))
        gp_sliding_window_model_wrappers_by_threshold = createSlidingWindowGPWrappers(train_data_single_marker_times,
            train_data_single_marker_single_coords, marker_num, coord_string, min_window_inc, window_inc_multipliers,
            window_size, thresholds)
        for threshold, window_inc, gp_sliding_window_model_wrapper in gp_sliding_window_model_wrappers_by_threshold:
            sliding_window_trajectory_prediction, sliding_window_trajectory_std_dev = \
                gp_sliding_window_model_wrapper.predictTrajectory(test_data_single_marker_times)
            #plotPredictionVsActual(test_data_single_marker_times, sliding_window_trajectory_prediction,
                # sliding_window_trajectory_std_dev, test_data_single_marker_single_coords, marker_num,
                # coord_string, train_data_single_marker_times, train_data_single_marker_single_coords)

            # Compute MSE for test data coords
            mean_square_error_sliding_hyperparams = computeTrajectoryError(test_data_single_marker_single_coords,
                                                                               sliding_window_trajectory_prediction)
            # print("Mean square error for global set of hyperparams " + str(mean_square_error_single_hyperparams))
            print("Mean square error for sliding window set of hyperparams for threshold " + str(threshold) + " and window inc " + str(window_inc) + ": " + str(mean_square_error_sliding_hyperparams))
            mse_by_window_config.append((window_size, window_inc, threshold, mean_square_error_sliding_hyperparams))

            if (mean_square_error_sliding_hyperparams < best_sliding_gp_model_mse):
                best_gp_model_wrapper = gp_sliding_window_model_wrapper
                best_sliding_gp_model_mse = mean_square_error_sliding_hyperparams

    # Uncomment to write data to file
    # new_fpath = "sliding_window_mses_marker_" + str(marker_num) + "_coord_" + coord_string + "_" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".pkl"
    # joblib.dump(mse_by_window_config, new_fpath)

    # Compute the prediction and the standard deviation for the sliding window approach and plot the results
    sliding_window_trajectory_pred, sliding_window_traj_std_dev = \
        best_gp_model_wrapper.predictTrajectory(test_data_single_marker_times)
    plotPredictionVsActual(test_data_single_marker_times, sliding_window_trajectory_pred,
                           sliding_window_traj_std_dev, test_data_single_marker_single_coords, marker_num,
                           coord_string, train_data_single_marker_times, train_data_single_marker_single_coords)

    hyperparam_results_for_sliding_window = []

    # Plot the hyperparameter results
    for i in range(len(best_gp_model_wrapper.sub_models)):
        submodel_data = best_gp_model_wrapper.sub_models[i]
        submodel_window_start = submodel_data[0]
        submodel_params = submodel_data[1].getHyperParams()

        if (i == 0):
            submodel_window_start = 0.0

        if (i == (len(best_gp_model_wrapper.sub_models) - 1)):
            window_end = train_data_single_marker_times[-1]
        else:
            window_end = best_gp_model_wrapper.sub_models[i + 1][0]
        hyperparam_results_for_sliding_window.append((submodel_window_start, window_end, submodel_params))

    # Plot the hyperparameters
    plotHyperparamsBySlidingWindow(hyperparam_results_for_sliding_window)

    print("Mean square error for global set of hyperparams " + str(mean_square_error_single_hyperparams))
    print("Mean squared error for sliding window hyperparams" + str(best_sliding_gp_model_mse))

def executeAssign3():

    """
    Execute the third assignment.
    """

    # Load the training and test data
    # train_data_file_1 = "data_GP/AG/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213203046-59968-right-speed_0.500.csv"
    # train_data_file_2 = "data_GP/AG/block2-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204004-59968-right-speed_0.500.csv"
    # train_data_file_3 = "data_GP/AG/block3-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204208-59968-right-speed_0.500.csv"
    # train_data_file_4 = "data_GP/AG/block4-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204925-59968-right-speed_0.500.csv"
    train_data_file_1 = "data_GP/CJ/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161207111012-59968-right-speed_0.500.csv"
    train_data_file_2 = "data_GP/CJ/block2-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161207111143-59968-right-speed_0.500.csv"
    train_data_file_3 = "data_GP/CJ/block3-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161207112226-59968-right-speed_0.500.csv"
    train_data_file_4 = "data_GP/CJ/block4-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161207112544-59968-right-speed_0.500.csv"
    train_data_files = [train_data_file_1, train_data_file_2, train_data_file_3, train_data_file_4]
    # test_data_file = "data_GP/AG/block5-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213210121-59968-right-speed_0.500.csv"
    test_data_file = "data_GP/CJ/block5-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161207113602-59968-right-speed_0.500.csv"

    train_data_sets = [readData(file_name) for file_name in train_data_files]
    test_data_set = readData(test_data_file)

    # marker = 9
    # coord = "z"

    # NOTE: Change the marker and coordinate configuration here
    marker = 37
    coord = "z"

    # Execute the assignment with the given configuration
    executeAssign3WithConfig(train_data_sets, test_data_set, marker, coord)

if __name__=="__main__":

    executeAssign3()
