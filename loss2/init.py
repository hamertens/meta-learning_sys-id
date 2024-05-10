import pandas as pd
import random
import numpy as np
from models_functions import mlp_model
from setup import DATA_FILEPATH, DATAFRAME_COLUMNS
from keras_uncertainty.models import DeepEnsembleRegressor


file_path_train_inputs = DATA_FILEPATH + 'train_inputs.csv'
df_train_inputs = pd.read_csv(file_path_train_inputs)
train_inputs = df_train_inputs.values

file_path_train_outputs = DATA_FILEPATH + 'train_outputs.csv'
df_train_outputs = pd.read_csv(file_path_train_outputs)
train_outputs = df_train_outputs.values

# Use random.randint() to generate a random index
# Generate the first random index
random_index1 = random.randint(0, len(train_inputs) - 1)

# Initialize the second random index to be the same as the first one
random_index2 = random_index1

# Keep generating a new random index until it's different from the first one
while random_index2 == random_index1:
    random_index2 = random.randint(0, len(train_inputs) - 1)

training_inputs = np.array([train_inputs[random_index1], train_inputs[random_index2]])
training_outputs = np.array([train_outputs[random_index1], train_outputs[random_index2]])

# Creating DataFrames for training inputs and outputs
training_inputs_active_df = pd.DataFrame(training_inputs, columns=DATAFRAME_COLUMNS)
training_outputs_active_df = pd.DataFrame(training_outputs, columns=DATAFRAME_COLUMNS)

# Saving DataFrames to CSV files
training_inputs_active_df.to_csv('output_data/training_inputs_active.csv', index=False)
training_outputs_active_df.to_csv('output_data/training_outputs_active.csv', index=False)

metric_df = pd.DataFrame({'RMSE': [], 'Variance': [], "Train_RMSE": []})
metric_df.to_csv("output_data/metrics.csv", index=False)


# Read DataFrame from CSV file
hyperparameters_df = pd.read_csv('output_data/hyperparameters.csv')

# Extract the values from the DataFrame
num_epochs = hyperparameters_df['epochs'][0]
#num_epochs = 500
batch_size = hyperparameters_df['batch_size'][0]
#batch_size = 32
# = hyperparameters_df['num_models'][0]
num_models = 5
# Train on current inputs and outputs
model = DeepEnsembleRegressor(mlp_model, num_models)
model.fit(training_inputs, training_outputs, epochs=num_epochs, batch_size = batch_size)

#model.save('output_data/ensemble_model')
model.save_weights('output_data/ensemble_model_weights')
model.save_weights_training('output_data/ensemble_model_weights_training')

# Predict for all possible input pairs
Y_pred, _ = model.predict(training_inputs)

predictions = Y_pred
predictions_df = pd.DataFrame(predictions, columns=DATAFRAME_COLUMNS)
predictions_df.to_csv('output_data/predictions.csv', index=False)