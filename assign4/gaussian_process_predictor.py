"""
Amanda Adkins
Fall 2020 CS391L Assignment 4

gaussian_process_predictor.py - Contains classes for making predictions for input timestamps using Gaussian Processes
"""
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Product, Sum, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np

class GaussianProcessHyperparams:
    """
    Contains setting for the RBF length, RBF variance, and noise
    """

    def __init__(self, rbf_len, rbf_variance, noise):
        """
        Create the hyperparameters object.

        Args:
            rbf_len (float):        Value for the RBF length hyperparameter.
            rbf_variance (float):   Value for the RBF variance hyperparameter.
            noise (float):          Value for the noise hyperparameter.
        """
        self.rbf_len = rbf_len
        self.rbf_variance = rbf_variance
        self.noise = noise

    def __str__(self):
        return "Hyperparams(RBF_len:" + str(self.rbf_len) + ", RBF_var:" + str(self.rbf_variance) + ", noise:" + str(self.noise) + ")"

class GaussianProcessModel:
    """
    Gaussian process model with a single set of hyperparameters.
    """

    def __init__(self, gp_hyperparams, gp_restarts, hyperparams_modifiable):
        """
        Initialize the Gaussian process model.

        Args:
            gp_hyperparams (GaussianProcessHyperparams):    Initial set of hyperparameters to use for the gaussian
                                                            process regressor.
            gp_restarts (int):                              Number of restarts to perform (used to mitigate against
                                                            local minima).
            hyperparams_modifiable (bool):                  True if the hyperparams should be optimized during fitting,
                                                            false otherwise.
        """
        if (hyperparams_modifiable):
            self.constant_kernel = ConstantKernel(gp_hyperparams.rbf_variance, (1e-10, 1e10))
            self.rbf_kernel = RBF(gp_hyperparams.rbf_len, (1e-10, 1e10))
            self.noise_kernel = WhiteKernel(gp_hyperparams.noise, (1e-10, 1e10))
        else:
            self.constant_kernel = ConstantKernel(gp_hyperparams.rbf_variance, "fixed")
            self.rbf_kernel = RBF(gp_hyperparams.rbf_len, "fixed")
            self.noise_kernel = WhiteKernel(gp_hyperparams.noise, "fixed")
        self.kernel = Sum(Product(self.constant_kernel, self.rbf_kernel), self.noise_kernel)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=gp_restarts, alpha=0.0)

    def trainModel(self, data_x, data_y):

        """
        Train the model. This assumes that the invalid data has already been filtered out.

        Args:
            data_x (2D Numpy array):    Input data. Each row is one feature. In this case, each row should be a single
                                        timestamp.
            data_y (2D Numpy array):    Output data. Each row is one output. In this case, each row should be a single
                                        coordinate value (for the coordinate that we're training on).
        """
        self.gp = self.gp.fit(data_x, data_y)

    def getHyperParams(self):
        """
        Get the hyperparameters from the trained model. Assumes that the model has already been trained.

        Returns:
            The hyperparameters from the trained model.
        """
        kernel_params = self.gp.kernel_.get_params()
        rbf_var = kernel_params['k1__k1__constant_value']
        rbf_len = kernel_params['k1__k2__length_scale']
        noise = kernel_params['k2__noise_level']
        hyperparams = GaussianProcessHyperparams(rbf_len, rbf_var, noise)
        return hyperparams

    def predictValue(self, timestamp):
        """
        Predict the output value for the given timestamp (input value).

        Args:
            timestamp (float): Timestamp to predict the coordinate value for.  TODO is this the right data type?
        Returns:
            Tuple of the predicted value and the standard deviation for the input timestamp.
        """
        timestamp_array = timestamp.reshape(-1, 1) # TODO (should this be reshaped differently)
        output = self.gp.predict(timestamp_array, return_std=True)
        return output


class GaussianProcessModelWithVaryingHyperparameters:

    """
    Gaussian process model that has different hyperparameters used depending on the timestamp.
    """

    def __init__(self, sub_models):
        """
        Constructor for the GaussianProcessModelWithVaryingHyperparameters

        Args:
            sub_models (List of tuples):    GaussianProcessModel objects with the time range that each should be used
                                            for. Each tuple contains the start time of the window that the model is
                                            applicable for and a GaussianProcessModel that should be used after the
                                            start time. The end time for the window is the start time of the next entry
                                            in the list (or if the tuple is the last one, any timestamps after the start
                                            time should use the last model). Due to this, the models are ordered by the
                                            order of the windows.
        """
        self.sub_models = sub_models

    def predictValue(self, timestamp):
        """
        Predict the output value for the given timestamp (input value).

        Args:
            timestamp (float): Timestamp to predict the coordinate value for.  TODO is this the right data type?
        Returns:
            Tuple of the predicted value and the standard deviation for the input timestamp.
        """

        for sub_model_index in range(len(self.sub_models)):
            if (sub_model_index >= (len(self.sub_models) - 1)):
                current_submodel = self.sub_models[sub_model_index]
                return current_submodel[1].predictValue(timestamp)
            else:
                next_submodel = self.sub_models[sub_model_index + 1]
                if (timestamp < next_submodel[0]):
                    current_submodel = self.sub_models[sub_model_index]
                    return current_submodel[1].predictValue(timestamp)

    def predictTrajectory(self, trajectory_timestamps):
        """
        Predict the output value for each timestamp in the trajectory_timestamp.

        Args:
            trajectory_timestamps (TODO data type?): List of timestamps to predict the coordinate value for.
        Returns:
            Tuple of 2 numpy arrays. First contains the predicted output values for the given timestamps, the second
            contains the standard deviations for each of the timestamps.
        """
        predictions = []
        sigmas = []
        for timestamp in trajectory_timestamps:
            prediction, sigma = self.predictValue(timestamp)
            predictions.append(prediction[0])
            sigmas.append(sigma[0])
        predictions = np.vstack(predictions)
        sigmas = np.vstack(sigmas)
        predictions = np.reshape(predictions, -1)
        sigmas = np.reshape(sigmas, -1)
        return (predictions, sigmas)


