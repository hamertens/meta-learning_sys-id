import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics import mean_squared_error
import os

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class GP:
    def __init__(self, training_type, kernel_type, sample_inputs, sample_outputs, test_inputs, test_outputs, training_inputs, training_outputs):
        self.training_type = training_type
        self.kernel_type = kernel_type
        self.output_data_dir = os.path.join(os.path.dirname(__file__), '..', 'output_data')
        self.file_path_parameters = os.path.join(self.output_data_dir, 'parameters.csv')

        self.sample_inputs = sample_inputs
        self.sample_outputs = sample_outputs
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        self.test_inputs = test_inputs
        self.test_outputs = test_outputs

        


    def initialize_kernel(self):
        #kernel = None
        self.parameters_df = pd.read_csv(self.file_path_parameters)
        if self.training_type == 'continuous':
            length_scale = self.parameters_df.iloc[-1]['length_scale']
        else:
            length_scale = 1.0

        if self.kernel_type == 'matern':
                kernel = Matern(length_scale=length_scale, nu=1.5)
        elif self.kernel_type == 'rbf':
                kernel = RBF(length_scale=length_scale)
        return kernel

    def update_parameters(self):
        learned_length_scale = self.gpr.kernel_.length_scale        
        new_row_parameters_df = pd.DataFrame({'length_scale': [learned_length_scale]})
        self.parameters_df = pd.concat([self.parameters_df, new_row_parameters_df], ignore_index=True)
        self.parameters_df.to_csv(self.file_path_parameters, index=False)

    def find_max_uncertainty(self):
            Y_pred, Y_std = self.gpr.predict(self.sample_inputs, return_std=True)
            sum_values = Y_std.sum(axis=1)
            max_index = np.argmax(sum_values)
            return max_index, Y_pred[max_index], np.sum(Y_std)
    
    def first_run(self):
        if self.kernel_type == 'matern':
            self.kernel = Matern(length_scale=1.0, nu=1.5)
        elif self.kernel_type == 'rbf':
            self.kernel = RBF(length_scale=1.0)
        self.gpr = GaussianProcessRegressor(kernel=self.kernel)
        self.gpr.fit(self.training_inputs, self.training_outputs)
        Y_pred = self.gpr.predict(self.training_inputs, return_std=False)

        learned_length_scale = self.gpr.kernel_.length_scale
        parameters_df = pd.DataFrame({'length_scale': [learned_length_scale]})
        parameters_df.to_csv(self.file_path_parameters, index=False)

        return Y_pred
    
    def active_learning(self):
        self.kernel = self.initialize_kernel()
        self.gpr = GaussianProcessRegressor(kernel=self.kernel)
        self.gpr.fit(self.training_inputs, self.training_outputs)
        max_index, new_prediction, variance = self.find_max_uncertainty()
        #Append RMSE and variance
        prediction = self.gpr.predict(self.test_inputs, return_std=False)
        # Append the new array to training inputs
        RMSE = np.sqrt(mean_squared_error(self.test_outputs, prediction))

        self.update_parameters()

        return max_index, new_prediction, variance, RMSE
        