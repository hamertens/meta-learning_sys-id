from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import pandas as pd
import subprocess


bash_script_path = "/home/hansm/active_learning/ICS_635_project/Ensemble_continuous/ensemble.sh"

def train_evaluate(learning_rate, num_layers, num_neurons, num_epochs, batch_size, num_models):

    # Create a DataFrame
    hyperparameters_data = {'lr': [learning_rate],
            'epochs': [num_epochs],
            'num_neurons': [num_neurons],
            'num_layers': [num_layers],
            'batch_size': [batch_size],
            'num_models': [num_models]}
    hyperparameters_df = pd.DataFrame(hyperparameters_data)


    # Save DataFrame to a CSV file
    hyperparameters_df.to_csv('output_data/hyperparameters.csv', index=False)
    subprocess.call([bash_script_path])

    # Read DataFrame from CSV file
    metric_df = pd.read_csv('output_data/metrics.csv')
    length = len(metric_df)
    
    return -length

param_space  = [Real(0.001, 0.05, name='learning_rate'), 
                Integer(1, 5, name='num_layers'), 
                Integer(16, 128, name='num_neurons'), 
                Integer(200, 1000, name='num_epochs'),
                Integer(16, 128, name='batch_size'),
                Integer(3, 8, name='num_models')]
# Perform Bayesian optimization
res = gp_minimize(
    lambda params: train_evaluate(*params),
    param_space,
    n_calls=40,
    random_state=42,
)

# Print the optimal parameters and minimum value found
print("Optimal parameters:", res.x)
print("Minimum value found:", res.fun)

# Save the optimal parameters to a CSV file
optimal_params = {'learning_rate': [res.x[0]],
                  'num_layers': [res.x[1]],
                  'num_neurons': [res.x[2]],
                  'num_epochs': [res.x[3]],
                  'batch_size': [res.x[4]],
                  'num_models': [res.x[5]]}
optimal_params_df = pd.DataFrame(optimal_params)
optimal_params_df.to_csv('output_data/optimal_hyperparameters.csv', index=False)
