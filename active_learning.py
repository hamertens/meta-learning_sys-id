import numpy as np
from systems import systems
from utils import functions
import sys 
import time
import argparse
import pandas as pd

start_time = time.time()

parser = argparse.ArgumentParser(description="Import dynamical system and model specifications")
parser.add_argument("-s", "--system", type=str, required=True, help="Type (required)")
parser.add_argument("-of", "--output_folder", type=str, required=True, help="Output Folder (required)")

args = parser.parse_args()
system_type = args.system
output_folder = args.output_folder


# Load the System
if system_type == "lorenz":
    system = systems.Lorenz()
elif system_type == "pendulum":
    system = systems.Pendulum()
elif system_type == "double_pendulum":
    system = systems.DoublePendulum()
elif system_type == "two_tank_system":
    system = systems.TwoTankSystem()
elif system_type == "actuated_pendulum":
    system = systems.ActuatedPendulum()

sample_inputs = system.load_sample_inputs()
sample_outputs = system.load_sample_outputs()
test_inputs = system.load_test_inputs()
test_outputs = system.load_test_outputs()
validation_inputs = system.load_validation_inputs()
validation_outputs = system.load_validation_outputs()

# Read the active learning training inputs, training outputs and predictions from the CSV files
training_inputs, training_outputs, predictions = functions.read_csv_files(output_folder)


hyperparameters_df = pd.read_csv(output_folder + '/hyperparameters.csv')
num_layers = hyperparameters_df['num_layers'][0]
num_neurons = hyperparameters_df['num_neurons'][0]
learning_rate = hyperparameters_df['learning_rate'][0]
num_models = hyperparameters_df['num_models'][0]
dropout_rate = hyperparameters_df['dropout_rate'][0]

from models import ensemble
training_type = "continuous"
hps = {'epochs': 10, 'batch_size': 32, 'lr': learning_rate, 'num_models': num_models, 'num_neurons': num_neurons, 'num_layers': num_layers}
model = ensemble.Ensemble(training_type, sample_inputs, sample_outputs, test_inputs, test_outputs, validation_inputs, validation_outputs, training_inputs, training_outputs, hps, output_folder)


max_index, new_prediction, variance, RMSE = model.active_learning()


# Append new prediction, training input and training output
predictions = np.vstack((predictions, new_prediction))
training_inputs = np.vstack((training_inputs, sample_inputs[max_index]))
training_outputs = np.vstack((training_outputs, sample_outputs[max_index]))

end_time = time.time()
elapsed_time = end_time - start_time

# Write the updated training inputs, training outputs, predictions and metrics to CSV files
functions.write_csv_files(training_inputs, training_outputs, system.dataframe_columns_input, system.dataframe_columns_output, RMSE, variance, predictions, elapsed_time, output_folder)

# check convergence
if len(predictions) >= 20:
    convergence = functions.check_settling_time(predictions, training_outputs)
    if convergence == True:
        sys.exit(1)