"""
Amanda Adkins
Fall 2020 CS391L Assignment 3
"""

from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Product, Sum, WhiteKernel

import argparse
import csv
from gaussian_process_predictor import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib import collections as mc

import datetime

import joblib

# TODO
# Play around with marker
# Play around with coordinates
# Play around with sliding window size
# Play around with sliding window increment
# Play around with threshold for grouping windows
# Determine if this should be predicting 3D pose or single coordinate



class SingleCoordInfo:
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




# def plotHyperParamsByFrame(sliding_window_hyperparams, num_frames):
#     # TODO
#     # Assuming sliding_window_hyperparams has structure (start index, hyperparams)
#     sliding_window_index = 0
#     hyperparams_for_frames = []
#     for i in range(num_frames):
#         if (sliding_window_index < (num_frames - 1)):
#             next_start_index = sliding_window_hyperparams[sliding_window_index + 1]
#             if (i >= next_start_index):
#                 sliding_window_index += 1
#         hyperparams_for_frames.append(sliding_window_hyperparams[sliding_window_index])

    # TODO plot hyperparams_for_frames against range(num_frames)
    # We may have multiple hyperparams that need to be split into separate lists


def plotTrain(timesteps, values):
    plt.plot(timesteps, values, 'bo')
    plt.show()


def plotPredictionVsActual(timesteps, predicted_values, predicted_std_dev, actual_values, marker_num, coord_string, train_timestamps, train_values):

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
    # plt.ylim(-10, 20)
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
    # plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.title("Predicted vs Actual position for " + coord_string + " coord of marker " + str(marker_num))
    plt.show()

    # TODO do we want the original data points?




def computeTrajectoryError(actual_trajectory, predicted_trajectory):
    # TODO need more params
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
    print("Plotting")
    # pose_seq = data_set.getPoseSequenceForMarkerNum(marker_num)
    # Not sure which of these I need, making them all now and will delete unnecessary ones later
    valid_poses = []
    valid_x = []
    valid_y = []
    valid_z = []
    for (timestamp, pose_data) in pose_seq:
        if (pose_data.is_valid):
            pose = pose_data.marker_pose
            valid_poses.append(pose)
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
    rbf_len_lines = []
    rbf_var_lines = []
    noise_lines = []
    colors = []

    for window_start, window_end, hyperparam_results in hyperparam_results_for_sliding_window:
        rbf_len = hyperparam_results.rbf_len
        rbf_var = hyperparam_results.rbf_variance
        noise = hyperparam_results.noise
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
    window_starts = [0.0]
    curr_window_rbf_sum = 0.0
    curr_window_rbf_count = 0.0

    for window_start, window_end, hyperparams in hyperparam_results:
        if (window_start < window_starts[-1]):
            continue
        if (curr_window_rbf_count == 0):
            curr_window_rbf_sum = hyperparams.rbf_len
            curr_window_rbf_count += 1
        else:
            if (abs(hyperparams.rbf_len - (curr_window_rbf_sum / curr_window_rbf_count)) > new_window_threshold):
                curr_window_rbf_count = 0
                window_starts.append(window_end)
            else:
                curr_window_rbf_count += 1
                curr_window_rbf_sum += hyperparams.rbf_len

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
    plotHyperparamsBySlidingWindow(plottable_hyperparam_results)

    return GaussianProcessModelWithVaryingHyperparameters(updated_models)





    # Retrain using new windows



def createSlidingWindowGPWrapper(train_data_single_marker_times, train_data_single_marker_single_coords,
                                 test_data_single_marker_times, test_data_single_marker_single_coords, marker_num, coord_string):

    # Assumes that train times and values are sorted by time

    # TODO tune these
    sliding_window_size = 5 # Should be time based increment
    sliding_window_inc = 0.5 # Should be time based increment
    min_time = train_data_single_marker_times[0][0]
    # print(min_time)
    max_time = train_data_single_marker_times[-1][0]
    # print(max_time)

    window_start = min_time
    window_end = min(max_time, window_start + sliding_window_size)
    continue_loop = True
    not_last_loop = True
    default_init_hyperparams = GaussianProcessHyperparams(1.0, 0.05, 0.01)
    hyperparam_results = []
    # print(train_data_single_marker_times)
    # print(min_time)
    # print(max_time)
    while (continue_loop):
        print("Window start, window end " + str(window_start) + ", " + str(window_end))
        times_in_range, values_in_range = getDataInTimeRange(train_data_single_marker_times,
            train_data_single_marker_single_coords, window_start, window_end)
        # print(times_in_range)
        # print("Times and values in range shape")
        # print(times_in_range.shape)
        # print(values_in_range.shape)

        sliding_window_gp = GaussianProcessModel(default_init_hyperparams, 5, True)
        sliding_window_gp.trainModel(times_in_range, values_in_range)


        hyperparam_results.append((window_start, window_end, sliding_window_gp.getHyperParams()))
        # print(hyperparam_results)
        print(str(hyperparam_results[-1][2]))
        continue_loop = not_last_loop
        window_start += sliding_window_inc
        window_end = window_start + sliding_window_size
        if (window_end > max_time):
            window_end = max_time
            not_last_loop = False

    # Pickle hyperparam results
    results_to_save = {'sliding_window_size': sliding_window_size, 'sliding_window_inc': sliding_window_inc,
                       'marker_num': marker_num, 'coord_string': coord_string, 'hyperparam_results': hyperparam_results}

    # Uncomment to write to file
    new_fpath = "sliding_window_results" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".pkl"
    joblib.dump(results_to_save, new_fpath)

    plotHyperparamsBySlidingWindow(hyperparam_results)
    # TODO choose where to draw boundaries

    # RBF difference threshold TODO have no idea if this is a good value
    rbf_difference_threshold = 1.0
    # TODO should plot hyperparams in here

    return chooseWindows(hyperparam_results, rbf_difference_threshold, train_data_single_marker_times, train_data_single_marker_single_coords)





def sortTimestampsAndValuesByTime(timestamps, values):
    # print("Before sorting")
    # print(timestamps.shape)
    # print(values.shape)
    # print(timestamps)
    # print(values)
    # Sort the training data by the time
    timestamps_and_values = np.hstack((timestamps, values))
    # print(timestamps_and_values.shape)
    # print(timestamps_and_values)
    sorted_timestamps_and_values = timestamps_and_values[(timestamps_and_values[:, 0]).argsort()]
    # print(sorted_timestamps_and_values)
    sorted_times = sorted_timestamps_and_values[:, 0]
    sorted_values = sorted_timestamps_and_values[:, 1]

    sorted_times = sorted_times.reshape(-1, 1)
    sorted_values = sorted_values.reshape(-1, 1)
    # print("Sorted")
    # print(sorted_times)
    # print(sorted_values)

    return (sorted_times, sorted_values)



def executeAssign3WithConfig(train_data_sets, test_data_set, marker_num, coord_string):
    # Train whole thing with one set of hyperparams
    # single_hyperparam_setting = trainGP(train_data_sets, marker_num, coord_string)

    train_full_3d_pose_for_marker = [data_set.getPoseSequenceForMarkerNum(marker_num) for data_set in train_data_sets]
    test_full_3d_pose_for_marker = test_data_set.getPoseSequenceForMarkerNum(marker_num)
    # plot test 3d pose for marker
    # TODO name
    # for i in range(len(train_full_3d_pose_for_marker)):
    #     plotMarkerPos(train_full_3d_pose_for_marker[i], marker_num, "Train sequence " + str(i))
    # plotMarkerPos(test_full_3d_pose_for_marker, marker_num, "Test data sequence")
    #
    # train_single_coord_pose_for_marker = []
    # test_single_coord_pose_for_marker = []


    # test_times = np.random.randint(1, 30, (10, 1))
    # test_values = np.random.randint(1, 30, (10, 1))
    # sortTimestampsAndValuesByTime(test_times, test_values)

    train_data_single_marker_times, train_data_single_marker_single_coords = convertMarkerPosSeqToGPForamt(train_data_sets, marker_num, coord_string)
    test_data_single_marker_times, test_data_single_marker_single_coords = convertMarkerPosSeqToGPForamt([test_data_set], marker_num, coord_string)

    train_data_single_marker_times, train_data_single_marker_single_coords = sortTimestampsAndValuesByTime(train_data_single_marker_times, train_data_single_marker_single_coords)



    # plotTrain(train_data_single_marker_times, train_data_single_marker_single_coords)

    # Train single GP model (same params for whole time)
    # gp_initial_hyperparams = GaussianProcessHyperparams(1, 0.5, 0.6)
    # global_params_gp_model = GaussianProcessModel(gp_initial_hyperparams, 5, True)
    # global_params_gp_model.trainModel(train_data_single_marker_times, train_data_single_marker_single_coords)
    #
    # # Get global hyperparams
    # gp_model_global_model_wrapper = GaussianProcessModelWithVaryingHyperparameters([(0.0, global_params_gp_model)])
    # trajectory_prediction, trajectory_std_dev = gp_model_global_model_wrapper.predictTrajectory(test_data_single_marker_times)
    #
    # # Plot prediction + var for test_data times
    # plotPredictionVsActual(test_data_single_marker_times, trajectory_prediction, trajectory_std_dev, test_data_single_marker_single_coords, marker_num, coord_string, train_data_single_marker_times, train_data_single_marker_single_coords)
    #
    # # Compute MSE for test data coords
    # mean_square_error_single_hyperparams = computeTrajectoryError(test_data_single_marker_single_coords, trajectory_prediction)
    # print("Mean square error for global set of hyperparams " + str(mean_square_error_single_hyperparams))

    gp_sliding_window_model_wrapper = createSlidingWindowGPWrapper(train_data_single_marker_times,
        train_data_single_marker_single_coords, test_data_single_marker_times, test_data_single_marker_single_coords, marker_num, coord_string)
    sliding_window_trajectory_prediction, sliding_window_trajectory_std_dev = \
        gp_sliding_window_model_wrapper.predictTrajectory(test_data_single_marker_times)
    plotPredictionVsActual(test_data_single_marker_times, sliding_window_trajectory_prediction,
                           sliding_window_trajectory_std_dev, test_data_single_marker_single_coords, marker_num,
                           coord_string, train_data_single_marker_times, train_data_single_marker_single_coords)

    # Compute MSE for test data coords
    mean_square_error_sliding_hyperparams = computeTrajectoryError(test_data_single_marker_single_coords,
                                                                   sliding_window_trajectory_prediction)
    # print("Mean square error for global set of hyperparams " + str(mean_square_error_single_hyperparams))
    print("Mean square error for sliding window set of hyperparams " + str(mean_square_error_sliding_hyperparams))








    # Get sliding window model
    # Plot prediction + var for test_data times
    # Compute MSE for test data coords
    # Plot param values




    # Train each sliding window with different hyperparams
    # Choose hyperparam boundaries
    # Plot mean and variance of globally trained model
    # Plot mean and variance of sliding window model
    # Show prediction for globally trained vs prediction for sliding window vs actual
    # Compute overall prediction error for globally trained vs overall prediction error for sliding window





def getCmdLineArgs():
    """
    Get the command line arguments.

    Returns:
        Arguments parsed from the command line input.
    """

    default_marker = 15
    default_coord = "z"


    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument()
    # TODO
    # arg_parser.add_argument("--sound_file_name", default=default_sound_file_name, help='Name of the file '\
        # 'containing the sound clips to mix and unmix.')
    # arg_parser.add_argument("--small_dataset_file_name", default=default_small_dataset_file_name, help='Name of the '\
        # 'containing the small data set and mixing matrix.')

    return arg_parser.parse_args()

def executeAssign3():

    # TODO revisit these settings
    train_data_file_1 = "data_GP/AG/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213203046-59968-right-speed_0.500.csv"
    train_data_file_2 = "data_GP/AG/block2-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204004-59968-right-speed_0.500.csv"
    train_data_file_3 = "data_GP/AG/block3-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204208-59968-right-speed_0.500.csv"
    train_data_file_4 = "data_GP/AG/block4-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204925-59968-right-speed_0.500.csv"
    train_data_files = [train_data_file_1, train_data_file_2, train_data_file_3, train_data_file_4]
    # train_data_files = [train_data_file_1, train_data_file_2, train_data_file_3]
    test_data_file = "data_GP/AG/block5-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213210121-59968-right-speed_0.500.csv"
    marker = 15
    coord = "z"

    train_data_sets = [readData(file_name) for file_name in train_data_files]
    test_data_set = readData(test_data_file)

    executeAssign3WithConfig(train_data_sets, test_data_set, marker, coord)

def readPickledResultsAndPlot():
    file_name = "sliding_window_results2020-10-27T23:05:50.pkl"

    results_dict = joblib.load(file_name)
    hyper_param_results = results_dict['hyperparam_results']
    plotHyperparamsBySlidingWindow(hyper_param_results)




if __name__=="__main__":

    parser_results = getCmdLineArgs()

    # executeAssign3Debug()
    executeAssign3()
    # readPickledResultsAndPlot()
