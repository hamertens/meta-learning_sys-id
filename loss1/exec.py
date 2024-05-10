import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from models_functions import check_settling_time, mlp_model
from setup import DATAFRAME_COLUMNS, DATA_FILEPATH, FOLDER_FILEPATH
from keras_uncertainty.models import DeepEnsembleRegressor
import sys
from keras.models import load_model
from keras_uncertainty.losses import regression_gaussian_nll_loss


file_path_train_inputs = DATA_FILEPATH + 'train_inputs.csv'
df_train_inputs = pd.read_csv(file_path_train_inputs)
train_inputs = df_train_inputs.values

file_path_train_outputs = DATA_FILEPATH + 'train_outputs.csv'
df_train_outputs = pd.read_csv(file_path_train_outputs)
train_outputs = df_train_outputs.values

file_path_test_inputs = DATA_FILEPATH + 'test_inputs.csv'
df_test_inputs = pd.read_csv(file_path_test_inputs)
test_inputs = df_test_inputs.values

file_path_test_outputs = DATA_FILEPATH + 'test_outputs.csv'
df_test_outputs = pd.read_csv(file_path_test_outputs)
test_outputs = df_test_outputs.values


file_path_training_inputs = FOLDER_FILEPATH + 'output_data/training_inputs_active.csv'
df_training_inputs = pd.read_csv(file_path_training_inputs)
training_inputs = df_training_inputs.values

file_path_training_outputs = FOLDER_FILEPATH + 'output_data/training_outputs_active.csv'
df_training_outputs = pd.read_csv(file_path_training_outputs)
training_outputs = df_training_outputs.values

file_path_predictions = FOLDER_FILEPATH + 'output_data/predictions.csv'
df_predictions = pd.read_csv(file_path_predictions)
model_predictions = df_predictions.values


# Read DataFrame from CSV file
hyperparameters_df = pd.read_csv('output_data/hyperparameters.csv')

# Extract the values from the DataFrame
num_epochs = hyperparameters_df['epochs'][0]
batch_size = hyperparameters_df['batch_size'][0]
learning_rate = hyperparameters_df['lr'][0]
#num_models = hyperparameters_df['num_models'][0]

num_models = 5
model = DeepEnsembleRegressor(mlp_model, num_models)
model.load_weights('output_data/ensemble_model_weights/')
model.load_weights_training('output_data/ensemble_model_weights_training/')


for test_model, train_model in zip(model.test_estimators, model.train_estimators):
    last_layer_name = test_model.layers[-1].name
    optimizer = {"class_name": "adam", "config": {"learning_rate": learning_rate}}
    train_model.compile(loss=regression_gaussian_nll_loss(test_model.get_layer(last_layer_name).output), optimizer=optimizer)


model.fit(training_inputs, training_outputs, epochs=num_epochs, batch_size = batch_size)

model.save_weights('output_data/ensemble_model_weights')
model.save_weights_training('output_data/ensemble_model_weights_training')

# Predict for all possible input pairs
Y_pred, Y_std = model.predict(train_inputs)
# Find max uncertainty
sum_values = Y_std.sum(axis=1)
max_index = np.argmax(sum_values)
# Append the new array to training inputs
training_inputs = np.vstack((training_inputs, train_inputs[max_index]))
# Append the new array to training outputs
training_outputs = np.vstack((training_outputs, train_outputs[max_index]))
# Calculate if convergence is met
model_predictions = np.vstack((model_predictions, Y_pred[max_index]))
if len(model_predictions) >= 20:
    convergence = check_settling_time(model_predictions, training_outputs)
    if convergence == True:
        sys.exit(1)
#Append RMSE and variance
prediction, _ = model.predict(test_inputs)
# Calculate RMSE between the ground truth values and the predictions
RMSE = np.sqrt(mean_squared_error(test_outputs, prediction))

variance= np.sum(Y_std)

# Calculate training rmse
train_rmse = np.sqrt(mean_squared_error(train_outputs, Y_pred))

# Creating DataFrames for training inputs and outputs
training_inputs_active_df = pd.DataFrame(training_inputs, columns= DATAFRAME_COLUMNS)
training_outputs_active_df = pd.DataFrame(training_outputs, columns= DATAFRAME_COLUMNS)


# Saving DataFrames to CSV files
training_inputs_active_df.to_csv('output_data/training_inputs_active.csv', index=False)
training_outputs_active_df.to_csv('output_data/training_outputs_active.csv', index=False)

metrics_df = pd.read_csv('output_data/metrics.csv')
new_row_metrics = pd.DataFrame({'RMSE': [RMSE], 'Variance': [variance], "Train_RMSE": [train_rmse]})
metrics_df = pd.concat([metrics_df, new_row_metrics], ignore_index=True)
# Save the updated DataFrame back to the same CSV file
metrics_df.to_csv('output_data/metrics.csv', index=False)

predictions_df = pd.DataFrame(model_predictions, columns= DATAFRAME_COLUMNS)
predictions_df.to_csv('output_data/predictions.csv', index=False)