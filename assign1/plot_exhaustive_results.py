from assign1 import *

def readCsvIntoKnnResults(csv_file_name):
    knn_results_list = []
    with open(csv_file_name) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        first_row = True
        for row in csv_reader:
            if (first_row):
                first_row = False
                continue
            row_as_nums = [float(row_entry) for row_entry in row]
            knn_results_list.append(KnnResults(row_as_nums[0], row_as_nums[1], row_as_nums[2], row_as_nums[3], row_as_nums[4]))
    return knn_results_list

def getResultsWithTrainSetSizes(training_set_sizes, all_knn_results):
    matching_results = {}
    for knn_result in all_knn_results:
        if (knn_result.num_training_samples in training_set_sizes):
            results_so_far = matching_results.get(knn_result.num_training_samples, [])
            results_so_far.append(knn_result)
            matching_results[knn_result.num_training_samples] = results_so_far
    return matching_results

def getResultsWithFeatureCounts(feature_counts, all_knn_results):
    matching_results = {}
    for knn_result in all_knn_results:
        if (knn_result.num_principal_components in feature_counts):
            results_so_far = matching_results.get(knn_result.num_principal_components, [])
            results_so_far.append(knn_result)
            matching_results[knn_result.num_principal_components] = results_so_far
    return matching_results

def getResultsWithKValues(k_values, all_knn_results):
    matching_results = {}
    for knn_result in all_knn_results:
        if (knn_result.num_nearest_neighbors in k_values):
            results_so_far = matching_results.get(knn_result.num_nearest_neighbors, [])
            results_so_far.append(knn_result)
            matching_results[knn_result.num_nearest_neighbors] = results_so_far
    return matching_results

def getAllKValues(k_nn_results):
    return sorted(list(set([knn_result.num_nearest_neighbors for knn_result in k_nn_results])))

def getAllFeatureValues(k_nn_results):
    return sorted(list(set([knn_result.num_principal_components for knn_result in k_nn_results])))


def getAllTrainingSetSizes(k_nn_results):
    return sorted(list(set([knn_result.num_training_samples for knn_result in k_nn_results])))

def plotKNumXAxisFeatureCountSeries(k_nn_results, feature_counts_to_plot, vis_colors, sample_count):
    nearest_neighbor_values = getAllKValues(k_nn_results)
    nearest_neighbor_values = [k for k in nearest_neighbor_values if k <= 50]

    accuracy_by_feature_count = {}
    color_by_feature_count = {}
    vis_color_index = 0
    for feature_count in feature_counts_to_plot:
        feature_count_results = getResultsWithFeatureCounts([feature_count], k_nn_results)[feature_count]
        results_by_neighbor_count = {}

        for result in feature_count_results:
            results_by_neighbor_count[result.num_nearest_neighbors] = result
        y_vals_for_feature_count = []
        for k_val in nearest_neighbor_values:
            y_vals_for_feature_count.append(results_by_neighbor_count[k_val].accuracy_rate)
        feature_count_str = str(feature_count) + " features"
        color_by_feature_count[feature_count_str] = vis_colors[vis_color_index]
        accuracy_by_feature_count[feature_count_str] = y_vals_for_feature_count
        vis_color_index = (vis_color_index + 1) % len(vis_colors)
    plotGraph(nearest_neighbor_values, accuracy_by_feature_count, "K Value", "Accuracy rate", "Accuracy rate by k value for " + str(sample_count) + " samples",color_by_feature_count)

def plotFeatureCountAccuracyTrainSetSeries(k_nn_results, training_set_sizes_to_plot, vis_colors, k_value):
    feature_values = getAllFeatureValues(k_nn_results)
    feature_values = [val for val in feature_values if (val < 200) and (val >=5)]

    accuracy_by_train_set_size = {}
    color_by_train_set_size = {}
    feature_vals_by_train_set_size = {}
    vis_color_index = 0
    for train_set_size in training_set_sizes_to_plot:
        train_set_size_results = getResultsWithTrainSetSizes([train_set_size], k_nn_results)[train_set_size]
        results_by_feature_count = {}

        for result in train_set_size_results:
            results_by_feature_count[result.num_principal_components] = result
        y_vals_for_train_set_size = []
        x_vals_for_train_set_size = []
        for feature_val in feature_values:
            print("Feature values" + str(feature_val))
            if (feature_val in results_by_feature_count.keys()):
                x_vals_for_train_set_size.append(feature_val)
                y_vals_for_train_set_size.append(results_by_feature_count[feature_val].accuracy_rate)
        train_sample_str = str(train_set_size) + " samples"
        color_by_train_set_size[train_sample_str] = vis_colors[vis_color_index]
        accuracy_by_train_set_size[train_sample_str] = y_vals_for_train_set_size
        feature_vals_by_train_set_size[train_sample_str] = x_vals_for_train_set_size
        vis_color_index = (vis_color_index + 1) % len(vis_colors)
    plotGraph(feature_vals_by_train_set_size, accuracy_by_train_set_size, "Feature Count", "Accuracy rate", "Accuracy rate by feature count for k=" + str(k_value), color_by_train_set_size)

def plotFeatureCountAccuracyTrainSetSeries(k_nn_results, training_set_sizes_to_plot, vis_colors, k_value):
    feature_values = getAllFeatureValues(k_nn_results)

    accuracy_by_train_set_size = {}
    color_by_train_set_size = {}
    feature_vals_by_train_set_size = {}
    vis_color_index = 0
    for train_set_size in training_set_sizes_to_plot:
        train_set_size_results = getResultsWithTrainSetSizes([train_set_size], k_nn_results)[train_set_size]
        results_by_feature_count = {}

        for result in train_set_size_results:
            results_by_feature_count[result.num_principal_components] = result
        y_vals_for_train_set_size = []
        x_vals_for_train_set_size = []
        for feature_val in feature_values:
            print("Feature values" + str(feature_val))
            if (feature_val in results_by_feature_count.keys()):
                x_vals_for_train_set_size.append(feature_val)
                y_vals_for_train_set_size.append(results_by_feature_count[feature_val].classification_time_per_sample)
        train_sample_str = str(train_set_size) + " samples"
        color_by_train_set_size[train_sample_str] = vis_colors[vis_color_index]
        accuracy_by_train_set_size[train_sample_str] = y_vals_for_train_set_size
        feature_vals_by_train_set_size[train_sample_str] = x_vals_for_train_set_size
        vis_color_index = (vis_color_index + 1) % len(vis_colors)
    plotGraph(feature_vals_by_train_set_size, accuracy_by_train_set_size, "Feature Count", "Classification time per sample (seconds)", "Classification time by feature count for k=" + str(k_value), color_by_train_set_size)

def plotTrainSetSizeAccuracyWithFeatureCounts(k_nn_results, feature_counts_to_plot, vis_colors, k_value):
    training_set_sizes = getAllTrainingSetSizes(k_nn_results)

    accuracy_by_feature_count = {}
    color_by_feature_count = {}
    train_set_size_by_feature_count = {}
    vis_color_index = 0
    for feature_count in feature_counts_to_plot:
        feature_count_results = getResultsWithFeatureCounts([feature_count], k_nn_results)[feature_count]
        results_by_train_set_size = {}

        for result in feature_count_results:
            results_by_train_set_size[result.num_training_samples] = result
        y_vals_for_feature_count = []
        x_vals_for_feature_count = []
        for train_size in training_set_sizes:
            if (train_size in results_by_train_set_size.keys()):
                x_vals_for_feature_count.append(train_size)
                y_vals_for_feature_count.append(results_by_train_set_size[train_size].accuracy_rate)
        feature_count_str = str(feature_count) + " features"
        color_by_feature_count[feature_count_str] = vis_colors[vis_color_index]
        accuracy_by_feature_count[feature_count_str] = y_vals_for_feature_count
        train_set_size_by_feature_count[feature_count_str] = x_vals_for_feature_count
        vis_color_index = (vis_color_index + 1) % len(vis_colors)
    plotGraph(train_set_size_by_feature_count, accuracy_by_feature_count, "Training Set Size", "Accuracy rate", "Accuracy rate by training set size for k=" + str(k_value), color_by_feature_count)


if __name__=="__main__":
    csv_file_name = "assign_1_results_2020-09-10T11:19:18.csv"

    knn_results_list = readCsvIntoKnnResults(csv_file_name)


    largest_train_set_results = getResultsWithTrainSetSizes([784], knn_results_list)[784]
    results_for_k_5 = getResultsWithKValues([5], knn_results_list)[5]
    # print(largest_train_set_results)

    # plotFeatureCountAccuracyTrainSetSeries(results_for_k_5, [200, 300, 400, 500, 600], ['bD-', 'gD-', 'rD-', 'cD-', 'mD-', 'yD-', 'bo-','go-', 'ro-', 'co-'], 5)
    plotKNumXAxisFeatureCountSeries(largest_train_set_results, [10, 25, 50, 75, 100, 150, 250, 300], ['bD-', 'gD-', 'rD-', 'cD-', 'mD-', 'yD-', 'bo-', 'go-', 'ro-', 'co-'], 784)

    # plotTrainSetSizeAccuracyWithFeatureCounts(results_for_k_5, [15, 25, 45, 75, 100, 250, 500, 784],['bD-', 'gD-', 'rD-', 'cD-', 'mD-', 'yD-', 'bo-','go-', 'ro-', 'co-'], 5)
