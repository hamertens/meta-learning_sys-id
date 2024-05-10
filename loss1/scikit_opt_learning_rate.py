from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import pandas as pd
import subprocess


bash_script_path = "/home/hansm/active_learning/ICS_635_project/Ensemble_continuous/ensemble.sh"
data_histories = pd.DataFrame(columns=['lr', 'epochs', 'batch_size', 'length'])
data_histories.to_csv('output_data/data_histories.csv', index=False)

def train_evaluate(learning_rate, num_epochs, batch_size):

    # Create a DataFrame
    hyperparameters_data = {'lr': [learning_rate],
            'epochs': [num_epochs],
            'batch_size': [batch_size]}

    hyperparameters_df = pd.DataFrame(hyperparameters_data)

    # Save DataFrame to a CSV file
    hyperparameters_df.to_csv('output_data/hyperparameters.csv', index=False)
    subprocess.call([bash_script_path])

    # Read DataFrame from CSV file
    metric_df = pd.read_csv('output_data/metrics.csv')
    length = len(metric_df)    

    data_histories = pd.read_csv('output_data/data_histories.csv')
    new_row = pd.DataFrame({'lr': [learning_rate], 'epochs': [num_epochs], 'batch_size': [batch_size], 'length': [length]})
    # check if length in new_row is smaller than all other lengths in data_histories
    if length < data_histories['length'].min():
        # save metric_df to best_metrics.csv
        metric_df.to_csv('output_data/best_metrics.csv', index=False)
        # save time.csv to best_time.csv
        time_df = pd.read_csv('output_data/time.csv')
        time_df.to_csv('output_data/best_time.csv', index=False)
    data_histories = pd.concat([data_histories, new_row], ignore_index=True)
    data_histories.to_csv('output_data/data_histories.csv', index=False)
    return length

param_space  = [Real(0.0005, 0.05, name='learning_rate'), 
                Integer(200, 1000, name='num_epochs'),
                Integer(16, 128, name='batch_size')]

# Perform Bayesian optimization
res = gp_minimize(
    lambda params: train_evaluate(*params),
    param_space,
    n_calls=50,
    random_state=42,
)

# Print the optimal parameters and minimum value found
print("Optimal parameters:", res.x)
print("Minimum value found:", res.fun)

# Save the optimal parameters to a CSV file
optimal_params = {'learning_rate': [res.x[0]], 'num_epochs': [res.x[1]], 'batch_size': [res.x[2]], 'length': [res.fun]}
optimal_params_df = pd.DataFrame(optimal_params)
optimal_params_df.to_csv('output_data/optimal_hyperparameters.csv', index=False)
