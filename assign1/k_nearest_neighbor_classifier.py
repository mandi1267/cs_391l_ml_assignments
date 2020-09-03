from classifier_interface import *
from functools import partial
from scipy import stats


class KNearestNeighborClassifier(Classifier):

    def __init__(self, num_neighbors):
        super(KNearestNeighborClassifier, self).__init__("KNN-" + str(num_neighbors))
        self.num_neighbors = num_neighbors

    def trainClassifier(self, training_data, training_labels):
        print("Training")
        self.training_data = training_data
        self.training_labels = training_labels
        print(self.training_data)
        print(self.training_labels)

    def classifySample(self, sample):
        print("HERE2")
        dist_func = partial(self.getDistBetweenSamples, sample)
        distance_from_each_train_sample = np.apply_along_axis(dist_func, 1, self.training_data)
        print("Dist vec")
        print(distance_from_each_train_sample)
        print(self.training_labels)

        combined_dist_and_label = np.column_stack((distance_from_each_train_sample, self.training_labels))
        print("Combined")
        print(combined_dist_and_label)

        sorted_dist_and_label = combined_dist_and_label[combined_dist_and_label[:, 0].argsort()]
        print("Sorted")
        print(sorted_dist_and_label)
        k_highest_labels = sorted_dist_and_label[:self.num_neighbors, 1:]
        print(k_highest_labels)
        print("Mode")
        print(stats.mode(k_highest_labels)[0][0][0])

        print("New line")

        return stats.mode(k_highest_labels)[0][0][0]

    def getDistBetweenSamples(self, sample_1, sample_2):
        print(sample_1)
        print(sample_2)
        distance = np.linalg.norm(sample_1 - sample_2)
        print("Dist " + str(distance))
        return distance
