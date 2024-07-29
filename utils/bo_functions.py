import subprocess
import os
import pandas as pd
bash_script_path = "al.sh"

# Initialize a global counter
iteration_counter = 0

def train_evaluate_architecture(num_neurons, num_layers):
    
    global iteration_counter
    # Increment the counter
    iteration_counter += 1
    output_folder = 'output_data_' + str(iteration_counter)
    previous_output_folder = 'output_data_' + str(iteration_counter - 1)
    # create output_data folder if necessary
    output_data_filepath = os.path.join(os.path.dirname(__file__),'..', output_folder)
    if not os.path.exists(output_data_filepath):
        os.makedirs(output_data_filepath)


    # Create a DataFrame
    hyperparameters_data = {'num_neurons': [num_neurons],
            'num_layers': [num_layers]}

    hyperparameters_df = pd.DataFrame(hyperparameters_data)

    # Save DataFrame to a CSV file
    hyperparameters_df.to_csv(output_folder + '/hyperparameters.csv', index=False)
    subprocess.call([bash_script_path, str(iteration_counter)])
    
    

    # Read DataFrame from CSV file
    metric_df = pd.read_csv(output_folder + '/metrics.csv')
    length = len(metric_df)
    accuracy =  metric_df["RMSE"].iloc[-1]

    if iteration_counter == 1:
        #create empty dataframe with columns num_neurons, num_layers, length
        data_histories = pd.DataFrame(columns=['num_neurons', 'num_layers', 'length', 'RMSE', 'rmse*length'])
    else:
        data_histories = pd.read_csv(previous_output_folder + '/data_histories.csv')
    new_row = pd.DataFrame({'num_neurons': [num_neurons], 'num_layers': [num_layers], 'length': [length], 'RMSE': [accuracy], 'rmse*length': [length*accuracy]})

    prev_data_histories_length = len(data_histories)
    # check if length in new_row is smaller than all other lengths in data_histories
    # only consider entries in data_histories that were not in the previous data_histories
    if length < data_histories[prev_data_histories_length:]['length'].min():
        # save metric_df to best_metrics.csv
        metric_df.to_csv(output_folder + '/best_metrics.csv', index=False)
        # save time.csv to best_time.csv
        time_df = pd.read_csv(output_folder + '/time.csv')
        time_df.to_csv(output_folder + '/best_time.csv', index=False)
        hyperparameters_df.to_csv(output_folder + '/optimal_hyperparameters.csv', index=False)
    data_histories = pd.concat([data_histories, new_row], ignore_index=True)
    data_histories.to_csv(output_folder + '/data_histories.csv', index=False)
    return length

def train_evaluate_learning_rate(learning_rate):

    
    global iteration_counter
    # Increment the counter
    iteration_counter += 1
    output_folder = 'output_data_' + str(iteration_counter)
    previous_output_folder = 'output_data_' + str(iteration_counter - 1)
    # create output_data folder if necessary
    output_data_filepath = os.path.join(os.path.dirname(__file__), output_folder)
    if not os.path.exists(output_data_filepath):
        os.makedirs(output_data_filepath)
    # Create a DataFrame
    hyperparameters_data = {'learning_rate': [learning_rate]}

    hyperparameters_df = pd.DataFrame(hyperparameters_data)

    # Save DataFrame to a CSV file
    hyperparameters_df.to_csv(output_folder + '/hyperparameters.csv', index=False)
    subprocess.call([bash_script_path])

    # Read DataFrame from CSV file
    metric_df = pd.read_csv(output_folder + '/metrics.csv')
    length = len(metric_df)

    if iteration_counter == 1:
        #create empty dataframe with columns learning_rate, length
        data_histories = pd.DataFrame(columns=['learning_rate', 'length'])
    else:
        data_histories = pd.read_csv(previous_output_folder + '/data_histories.csv')

    
    new_row = pd.DataFrame({'learning_rate': [learning_rate]})
    prev_data_histories_length = len(data_histories)

    # check if length in new_row is smaller than all other lengths in data_histories
    # only consider entries in data_histories that were not in the previous data_histories
    if length < data_histories[prev_data_histories_length:]['length'].min():
        # save metric_df to best_metrics.csv
        metric_df.to_csv(output_folder + '/best_metrics.csv', index=False)
        # save time.csv to best_time.csv
        time_df = pd.read_csv(output_folder + '/time.csv')
        time_df.to_csv(output_folder + '/best_time.csv', index=False)
        hyperparameters_df.to_csv(output_folder + '/optimal_hyperparameters.csv', index=False)
    data_histories = pd.concat([data_histories, new_row], ignore_index=True)
    data_histories.to_csv(output_folder + '/data_histories.csv', index=False)
    return length


def train_evaluate_hyperparameters(hyperparameters):
    global iteration_counter
    iteration_counter += 1
    output_folder = f'output_data_{iteration_counter}'
    previous_output_folder = f'output_data_{iteration_counter - 1}'

    
    # Create output folder if it doesn't exist
    output_data_filepath = os.path.join(os.path.dirname(__file__),'..', output_folder)
    os.makedirs(output_data_filepath, exist_ok=True)
    
    # Default hyperparameter values
    default_values = {
        'num_neurons': 60,
        'num_layers': 2,
        'learning_rate': 0.001,
        'num_models': 5,
        'dropout_rate': 0.0
    }

    # Ensure all columns are present in the DataFrame
    hyperparameters_with_defaults = {**default_values, **hyperparameters}
    hyperparameters_df = pd.DataFrame([hyperparameters_with_defaults])
    hyperparameters_df.to_csv(os.path.join(output_folder, 'hyperparameters.csv'), index=False)
    
    # Run the bash script
    subprocess.call([bash_script_path, str(iteration_counter)])
    
    # Read the metrics from the generated file
    metric_df = pd.read_csv(os.path.join(output_folder, 'metrics.csv'))
    length = len(metric_df)
    accuracy = metric_df['RMSE'].iloc[-1]
    
    # Handle the data_histories file
    data_histories_path = os.path.join(previous_output_folder, 'data_histories.csv')
    if iteration_counter == 1 or not os.path.exists(data_histories_path):
        data_histories = pd.DataFrame(columns=list(hyperparameters_with_defaults.keys()) + ['length', 'RMSE', 'rmse*length'])
    else:
        data_histories = pd.read_csv(data_histories_path)
    
    new_row = pd.DataFrame({**hyperparameters_with_defaults, 'length': length, 'accuracy': accuracy, 'length*accuracy': length*accuracy}, index=[0])
    prev_data_histories_length = len(data_histories)

    if length < data_histories.iloc[prev_data_histories_length:]['length'].min():
        metric_df.to_csv(os.path.join(output_folder, 'best_metrics.csv'), index=False)
        time_df = pd.read_csv(os.path.join(output_folder, 'time.csv'))
        time_df.to_csv(os.path.join(output_folder, 'best_time.csv'), index=False)
        hyperparameters_df.to_csv(os.path.join(output_folder, 'optimal_hyperparameters.csv'), index=False)
    
    
    data_histories = pd.concat([data_histories, new_row], ignore_index=True)
    data_histories.to_csv(os.path.join(output_folder, 'data_histories.csv'), index=False)
    
    return length