
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Product, Sum, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np

np.random.seed(1)

class GaussianProcessHyperparams:

    def __init__(self, rbf_len, rbf_variance, noise):
        self.rbf_len = rbf_len
        self.rbf_variance = rbf_variance
        self.noise = noise

    def __str__(self):
        return "Hyperparams(RBF_len:" + str(self.rbf_len) + ", RBF_var:" + str(self.rbf_variance) + ", noise:" + str(self.noise) + ")"

class GaussianProcessModel:

    def __init__(self, gp_hyperparams, gp_restarts, hyperparams_modifiable):
        if (hyperparams_modifiable):
            self.constant_kernel = ConstantKernel(gp_hyperparams.rbf_variance, (1e-10, 1e10))
            self.rbf_kernel = RBF(gp_hyperparams.rbf_len, (1e-10, 1e10))
            self.noise_kernel = WhiteKernel(gp_hyperparams.noise, (1e-10, 1e10)) # TODO should this be white kernel or constant kernel?
        else:
            self.constant_kernel = ConstantKernel(gp_hyperparams.rbf_variance, "fixed")
            self.rbf_kernel = RBF(gp_hyperparams.rbf_len, "fixed")
            self.noise_kernel = WhiteKernel(gp_hyperparams.noise, "fixed")
        self.kernel = Sum(Product(self.constant_kernel, self.rbf_kernel), self.noise_kernel) # TODO should this be white kernel or constant kernel
        print("Before training")
        print(self.kernel)
        # self.gp = GaussianProcessRegressor(kernel = self.kernel, n_restarts_optimizer=gp_restarts, alpha=0.0).fit(x_data, y_data)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=gp_restarts, alpha=0.0) # TODO what should alpha be?

    def trainModel(self, data_x, data_y):
        # Assumes we've already filtered out invalid points
        # print(self.gp.kernel_.get_params())
        print("Training")
        self.gp = self.gp.fit(data_x, data_y)
        print(self.gp.kernel_.get_params())
        print("Old kernel")
        print(self.kernel)
        print(self.gp.kernel)
        print("New kernel")
        print(self.gp.kernel_)
        print("Done training")


    def getHyperParams(self):
        kernel_params = self.gp.kernel_.get_params()
        # print(kernel_params)
        rbf_var = kernel_params['k1__k1__constant_value']
        rbf_len = kernel_params['k1__k2__length_scale']
        # noise = kernel_params['k2__constant_value'] # If using constant kernel
        noise = kernel_params['k2__noise_level'] # If using white kernel
        hyperparams = GaussianProcessHyperparams(rbf_len, rbf_var, noise)
        return hyperparams

    def predictValue(self, timestamp):
        timestamp_array = timestamp.reshape(-1, 1)
        output = self.gp.predict(timestamp_array, return_std=True)
        # print("Result for time " + str(timestamp))
        # print(output)
        return output


class GaussianProcessModelWithVaryingHyperparameters:

    def __init__(self, sub_models):
        self.sub_models = sub_models

    def predictValue(self, timestamp):
        sub_model_index = 0
        for i in range(len(self.sub_models)):
            if (sub_model_index >= (len(self.sub_models) - 1)):
                current_submodel = self.sub_models[sub_model_index]
                return current_submodel[1].predictValue(timestamp)
            else:
                next_submodel = self.sub_models[sub_model_index + 1]
                if (timestamp < next_submodel[0]):
                    current_submodel = self.sub_models[sub_model_index]
                    return current_submodel[1].predictValue(timestamp)
                else:
                    sub_model_index += 1

    def predictTrajectory(self, trajectory_timestamps):
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
        print("Predictions")
        print(predictions.shape)
        print("Sigmas")
        print(sigmas.shape)
        return (predictions, sigmas)


