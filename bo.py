from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import pandas as pd
from utils.bo_functions import train_evaluate_architecture
from utils.bo_functions import train_evaluate_learning_rate



optimization_type = 'architecture'

if optimization_type == 'architecture':
    optimization_function = train_evaluate_architecture
    param_space  = [Integer(40, 200, name='num_neurons'),
                Integer(1, 2, name='num_layers')]
elif optimization_type == 'learning_rate':
    optimization_function = train_evaluate_learning_rate
    param_space  = [Real(0.0001, 0.001, name='learning_rate')]


# Perform Bayesian optimization with pretraining
res = gp_minimize(
    lambda params: optimization_function(*params),
    param_space,
    acq_func="LCB",
    #x0=x0,  # Use known parameter combinations as initial points
    #y0=y0,  # Corresponding objective values
    n_calls=10,  # Total number of calls to the objective function
    random_state=42
)