from classifier_interface import *
from functools import partial
from scipy import stats


class KNearestNeighborClassifier(Classifier):

    def __init__(self, num_neighbors, num_features):
        super(KNearestNeighborClassifier, self).__init__("K-NN_k=" + str(num_neighbors) + "_numFeat="+ str(num_features))
        self.num_neighbors = num_neighbors

    def trainClassifier(self, training_data, training_labels):
        # Store training data so each sample corresponds to a row
        self.training_data = np.transpose(training_data)
        self.training_labels = training_labels

    def classifySample(self, sample):
        # print("HERE2")
        # print("Num neighbors")
        # print(self.num_neighbors)
        dist_func = partial(self.getDistBetweenSamples, sample)
        distance_from_each_train_sample = np.apply_along_axis(dist_func, 1, self.training_data)
        # print("Dist vec")
        # print(distance_from_each_train_sample)
        # print(self.training_labels)

        combined_dist_and_label = np.column_stack((distance_from_each_train_sample, self.training_labels))
        # print("Combined")
        # print(combined_dist_and_label)

        sorted_dist_and_label = combined_dist_and_label[(-1 * combined_dist_and_label[:, 0]).argsort()]
        # print("Sorted")
        # print(sorted_dist_and_label)
        # TODO is this sorting correctly (or do we need to take the k last numbers)
        k_highest_labels = sorted_dist_and_label[:self.num_neighbors, 1:]
        # print(k_highest_labels)
        # print("Mode")
        # print(stats.mode(k_highest_labels)[1])
        # print(stats.mode(k_highest_labels)[0][0][0])

        return self.getUniqueMode(k_highest_labels)

    def getDistBetweenSamples(self, sample_1, sample_2):
        distance = np.linalg.norm(sample_1 - sample_2)
        return distance

    def getUniqueMode(self, k_highest_labels):
        """
        Get a unique mode from the labels matching the k closest neighbors.
        If there are multiple modes in the k closest neighbors, we'll break the tie by picking the modal value that
        occurs most in the L closest labels (where L is initially k and is reduced by 1 until there is a unique mode)

        TODO

        """

        count_dict = {}
        for label in k_highest_labels:
            # print(label)
            count_dict[label[0]] = count_dict.get(label[0], 0) + 1
        # print("Count dict")
        # print(count_dict)

        max_count = stats.mode(k_highest_labels)[1][0][0]
        modal_labels = [label for label, label_count in count_dict.items() if label_count == max_count]
        for label in k_highest_labels[::-1]:
            if (len(modal_labels) <= 1):
                break
            nth_highest_label = label[0]
            if (nth_highest_label in modal_labels):
                modal_labels.remove(nth_highest_label)

        # print("Assigned label")
        # print(modal_labels[0])

        return modal_labels[0]
