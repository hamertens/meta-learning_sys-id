from skopt import gp_minimize
from skopt.space import Integer
import pandas as pd
import subprocess

# Load the data from data_histories.csv
data_histories = pd.read_csv('output_data/data_histories.csv')

# Filter the data to include only rows where num_layers is 1 or 2
data_histories_filtered = data_histories[data_histories['num_layers'].isin([1, 2])]

# Extract parameter combinations and objective values
x0 = data_histories_filtered[['num_layers', 'num_neurons']].values
x0 = x0.tolist()
y0 = data_histories_filtered['loss'].values
y0 = y0.tolist()
#x0 = [[1, 16], [2, 156]]
#y0 = [100, 200]

bash_script_path = "/home/hansm/active_learning/ICS_635_project/Ensemble_cont_acc_acquisition/ensemble.sh"


def train_evaluate(num_layers, num_neurons):

    # Create a DataFrame
    hyperparameters_data = {'num_neurons': [num_neurons],
            'num_layers': [num_layers]}
    hyperparameters_df = pd.DataFrame(hyperparameters_data)

    # Save DataFrame to a CSV file
    hyperparameters_df.to_csv('output_data/hyperparameters.csv', index=False)
    subprocess.call([bash_script_path])

    # Read DataFrame from CSV file
    metric_df = pd.read_csv('output_data/metrics.csv')
    length = len(metric_df)    
    accuracy =  metric_df["RMSE"].iloc[-1]
    loss = length*accuracy

    data_histories = pd.read_csv('output_data/data_histories.csv')
    new_row = pd.DataFrame({'num_layers': [num_layers], 'num_neurons': [num_neurons], "length": [length], "RMSE": [accuracy], "loss": [loss]})
    # check if loss in new_row is smaller than all other lengths in data_histories
    if loss < data_histories['loss'].min():
        # save metric_df to best_metrics.csv
        metric_df.to_csv('output_data/best_metrics.csv', index=False)
        # save time.csv to best_time.csv
        time_df = pd.read_csv('output_data/time.csv')
        time_df.to_csv('output_data/best_time.csv', index=False)
    data_histories = pd.concat([data_histories, new_row], ignore_index=True)
    data_histories.to_csv('output_data/data_histories.csv', index=False)

    # get last entry in metrc_df["RMSE"]
    accuracy =  metric_df["RMSE"].iloc[-1]
    return length*accuracy


# Define the parameter space
param_space = [Integer(1, 2, name='num_layers'), 
               Integer(16, 200, name='num_neurons')]

# Perform Bayesian optimization with pretraining
res = gp_minimize(
    lambda params: train_evaluate(*params),
    param_space,
    x0=x0,  # Use known parameter combinations as initial points
    y0=y0,  # Corresponding objective values
    n_calls=50,  # Total number of calls to the objective function
    random_state=42
)

# Print the optimal parameters and minimum value found
print("Optimal parameters:", res.x)
print("Minimum value found:", res.fun)

# Save the optimal parameters to a CSV file
optimal_params = {'num_layers': [res.x[0]],
                  'num_neurons': [res.x[1]]}
optimal_params_df = pd.DataFrame(optimal_params)
optimal_params_df.to_csv('output_data/optimal_hyperparameters.csv', index=False)
