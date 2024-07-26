from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import pandas as pd
from utils.bo_functions import train_evaluate_architecture
from utils.bo_functions import train_evaluate_learning_rate
from utils.bo_functions import train_evaluate_hyperparameters



optimization_type = 'architecture'

if optimization_type == 'architecture':
    param_space = [Integer(40, 200, name='num_neurons'),
                   Integer(1, 2, name='num_layers')]
    optimization_function = lambda params: train_evaluate_hyperparameters({
        'num_neurons': params[0],
        'num_layers': params[1]
    })
elif optimization_type == 'learning_rate':
    param_space = [Real(0.0001, 0.001, name='learning_rate')]
    optimization_function = lambda params: train_evaluate_hyperparameters({
        'learning_rate': params[0]
    })

# Perform Bayesian optimization with pretraining
res = gp_minimize(
    optimization_function,
    param_space,
    acq_func="LCB",
    n_calls=10,
    random_state=42
)