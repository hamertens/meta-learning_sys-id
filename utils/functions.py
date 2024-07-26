import numpy as np
import pandas as pd

def check_settling_time(prediction, goal):
    # Ensure both arrays have at least 20 entries
    if len(prediction) < 20 or len(goal) < 20:
        raise ValueError("Both arrays should have at least 20 entries.")

    # Take the last 20 entries from both arrays
    last_20_prediction = prediction[-20:]
    last_20_goal = goal[-20:]

    # Calculate the maximum allowed average error (2%)
    max_average_error = 0.02

    # Check if the average error across all dimensions is within the 2% error band
    for val1, val2 in zip(last_20_prediction, last_20_goal):

        errors = np.abs((val1 - val2) / val2)  # Calculate the percentage error for each dimension
        avg_error = np.mean(errors)  # Calculate the average error across all dimensions
        if avg_error > max_average_error:
            return False  # Average error exceeds the 2% limit

    return True  # Average error is within the 2% limit for all 20 entries

def write_csv_files(training_inputs, training_outputs, input_columns, output_columns, RMSE, variance, predictions, elapsed_time, output_folder):
    # Creating DataFrames for training inputs and outputs
    training_inputs_active_df = pd.DataFrame(training_inputs, columns=input_columns)
    training_outputs_active_df = pd.DataFrame(training_outputs, columns=output_columns)

    training_inputs_active_df.to_csv(output_folder + '/training_inputs_active.csv', index=False)
    training_outputs_active_df.to_csv(output_folder + '/training_outputs_active.csv', index=False)

    predictions_df = pd.DataFrame(predictions, columns=output_columns)
    predictions_df.to_csv(output_folder + '/predictions.csv', index=False)

    metrics_df = pd.read_csv(output_folder + '/metrics.csv')
    new_row_metrics = pd.DataFrame({'RMSE': [RMSE], 'Variance': [variance], "Time": [elapsed_time]})
    metrics_df = pd.concat([metrics_df, new_row_metrics], ignore_index=True)
    # Save the updated DataFrame back to the same CSV file

    metrics_df.to_csv(output_folder +'/metrics.csv', index=False)


def read_csv_files(output_folder):
    
    file_path_training_inputs = output_folder + '/training_inputs_active.csv'
    df_training_inputs = pd.read_csv(file_path_training_inputs)
    training_inputs = df_training_inputs.values

    file_path_training_outputs = output_folder + '/training_outputs_active.csv'
    df_training_outputs = pd.read_csv(file_path_training_outputs)
    training_outputs = df_training_outputs.values

    file_path_predictions = output_folder + '/predictions.csv'
    df_predictions = pd.read_csv(file_path_predictions)
    predictions = df_predictions.values

    return training_inputs, training_outputs, predictions
