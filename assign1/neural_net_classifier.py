import tensorflow as tf
from classifier_interface import *
from display_utils import *

from functools import partial
import numpy as np
import time

def normalizeImageDataForNeuralNet(raw_image_data):
    """
    Normalize the image data to the 0-1 range.
    """
    return raw_image_data / 255.0

class NeuralNetClassifier(Classifier):
    """
    Classifier that uses a 2-layer neural network to classify images.
    """

    def __init__(self):
        super(NeuralNetClassifier, self).__init__("Neural Net Classifier")

    def trainClassifier(self, training_data, training_labels):
        """
        Train the classifier.

        Args:
            training_data (2D numpy array):     2D array of size P x L containing the training data, where each sample
                                                is a column.
            training_labels (1D numpy array):   1D array of length L containing the labels for the training data.
        """
        # Construct neural net with settings described here: https://www.tensorflow.org/tutorials/quickstart/beginner
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        training_data_as_rows = np.transpose(training_data)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
        self.model.fit(training_data_as_rows, training_labels, epochs=10)
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    def classifySample(self, sample):
        """
        Classify the given sample.

        Args:
            sample (1D numpy array): Sample to classify.
        Returns:
            Label for the sample.
        """
        reshaped_sample = np.reshape(sample, (1, -1))
        predictions = self.probability_model.predict(reshaped_sample)[0]
        return np.argmax(predictions)
