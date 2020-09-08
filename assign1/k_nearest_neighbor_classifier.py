from classifier_interface import *
from functools import partial
from scipy import stats


class KNearestNeighborClassifier(Classifier):
    """
    Classifier that finds the K nearest training samples to the sample to classify and returns the modal label from set
    labels for those K closest samples.
    """

    def __init__(self, num_neighbors, num_features):
        """
        Constructor for the classifier.

        Args:
            num_neighbors (int):    K in the K-nn algorithm. Provides the number of closest samples from which to get
                                    the label for a sample to classify.
            num_features (int):     Number of features that are used in the classification. Used only for naming the
                                    classifier.
                                    # TODO can we remove the number of features?
        """
        super(KNearestNeighborClassifier, self).__init__("K-NN_k=" + str(num_neighbors) + "_numFeat="+ str(num_features))
        self.num_neighbors = num_neighbors

    def trainClassifier(self, training_data, training_labels):
        """
        Train the classifier.

        Args:
            training_data (2D numpy array):     2D array of size P x L containing the training data, where each sample
                                                is a column.
            training_labels (1D numpy array):   1D array of length L containing the labels for the training data.
        """

        # Store training data so each sample corresponds to a row
        self.training_data = np.transpose(training_data)
        self.training_labels = training_labels

    def classifySample(self, sample):
        """
        Classify the given sample.

        Args:
            sample (1D numpy array): Sample to classify.
        Returns:
            Label for the sample.
        """

        # Create a vector that contains the distance of the sample from each training sample (with the nth value in the
        # distance vector corresponding to the distance from the nth training sample).
        dist_func = partial(self.getDistBetweenSamples, sample)
        distance_from_each_train_sample = np.apply_along_axis(dist_func, 1, self.training_data)

        # Merge the distance vector and label vectors so that each row contains the distance and label for a training
        # sample, then sort by the distance to get the labels of the K closest training samples
        combined_dist_and_label = np.column_stack((distance_from_each_train_sample, self.training_labels))
        sorted_dist_and_label = combined_dist_and_label[(combined_dist_and_label[:, 0]).argsort()]

        # Get the first labels of the k closest training points
        # List is sorted in ascending order, so these will be the first k entries
        k_highest_labels = sorted_dist_and_label[:self.num_neighbors, 1:]

        # Get the mode from the labels. This function resolves ties by reducing the K value by 1 until there is a mode.
        return self.getUniqueMode(k_highest_labels)

    def getDistBetweenSamples(self, sample_1, sample_2):
        """
        Get the distance between the two given samples.

        Args:
            sample_1 (1D numpy array): First sample to use in distance calculation.
            sample_2 (1D numpy array): Second sample to use in distance calculation.
        Returns:
            Euclidean distance between the samples.
        """
        distance = np.linalg.norm(sample_1 - sample_2)
        return distance

    def getUniqueMode(self, k_highest_labels):
        """
        Get a unique mode from the labels matching the k closest neighbors.
        If there are multiple modes in the k closest neighbors, we'll break the tie by picking the modal value that
        occurs most in the L closest labels (where L is initially k and is reduced by 1 until there is a unique mode).

        Args:
            k_highest_labels (1D numpy array):  Numpy array containing the labels corresponding to the k closest
                                                training samples to the sample that is being classified. The labels are
                                                sorted such that the first label corresonds to the closest training
                                                sample and the kth label corresponds to the kth closest training sample.
        Returns:
            The modal label. If there is a unique mode, it is returned. If there are multiple modes in the set of labels,
            the label corresponding to the most distant training sample will be removed until the tie among the modal
            values is broken.
        """

        # Get the count of each label in the set of k highest labels
        count_dict = {}
        for label in k_highest_labels:
            count_dict[label[0]] = count_dict.get(label[0], 0) + 1

        # Find the number of occurrences of the modal value and find all labels that have that number of occurrences
        # (get all modal labels)
        max_count = stats.mode(k_highest_labels)[1][0][0]
        modal_labels = [label for label, label_count in count_dict.items() if label_count == max_count]

        # While there are multiple modal labels, take the label from the farthest training sample out of consideration
        # In other words, break ties among modes by removing labels from the end of the vector. Never considers values
        # that aren't in the initial set of modes.
        for label in k_highest_labels[::-1]:
            if (len(modal_labels) <= 1):
                break
            nth_highest_label = label[0]
            if (nth_highest_label in modal_labels):
                modal_labels.remove(nth_highest_label)

        # Return the modal label (after breaking any ties)
        return modal_labels[0]
