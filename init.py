import pandas as pd
import random
import numpy as np
from systems import systems
import os
import argparse

parser = argparse.ArgumentParser(description="Import dynamical system and model specifications")
parser.add_argument("-s", "--system", type=str, required=True, help="System (required)")
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

# Load the System
sample_inputs = system.load_sample_inputs()
sample_outputs = system.load_sample_outputs()
test_inputs = system.load_test_inputs()
test_outputs = system.load_test_outputs()
validation_inputs = system.load_validation_inputs()
validation_outputs = system.load_validation_outputs()

# Use random.randint() to generate a random index
# Generate the first random index
random_index1 = random.randint(0, len(sample_inputs) - 1)
# Initialize the second random index to be the same as the first one
random_index2 = random_index1
# Keep generating a new random index until it's different from the first one
while random_index2 == random_index1:
    random_index2 = random.randint(0, len(sample_inputs) - 1)

training_inputs = np.array([sample_inputs[random_index1], sample_inputs[random_index2]])
training_outputs = np.array([sample_outputs[random_index1], sample_outputs[random_index2]])

# Creating DataFrames for training inputs and outputs
training_inputs_active_df = pd.DataFrame(training_inputs, columns=system.dataframe_columns_input)
training_outputs_active_df = pd.DataFrame(training_outputs, columns=system.dataframe_columns_output)

# create output_data folder if necessary
output_data_filepath = os.path.join(os.path.dirname(__file__), output_folder)
if not os.path.exists(output_data_filepath):
    os.makedirs(output_data_filepath)


# Saving DataFrames to CSV files
training_inputs_active_df.to_csv(output_folder + '/training_inputs_active.csv', index=False)
training_outputs_active_df.to_csv(output_folder + '/training_outputs_active.csv', index=False)

metric_df = pd.DataFrame({'RMSE': [], 'Variance': [], 'Time': []})
metric_df.to_csv(output_folder + "/metrics.csv", index=False)

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


predictions = model.first_run()
predictions_df = pd.DataFrame(predictions, columns=system.dataframe_columns_output)
predictions_df.to_csv(output_folder + '/predictions.csv', index=False)

data_histories = pd.DataFrame(columns=['num_neurons', 'num_layers', 'length'])
data_histories.to_csv(output_folder + '/data_histories.csv', index=False)
