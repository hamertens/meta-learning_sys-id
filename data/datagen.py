import numpy as np
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
from itertools import product
import math
import pandas as pd
import random

# Define the Lorenz system ODE function
def lorenz_system(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Set the parameters
sigma = 10
rho = 28
beta = 8/3

def dataset_gen(initials, t):
  initials = initials
  # Solve the ODE using odeint
  solution = odeint(lorenz_system, initials, t, args=(sigma, rho, beta))
  # Extract the individual trajectories
  x_trajectory, y_trajectory, z_trajectory = solution[:, 0], solution[:, 1], solution[:, 2]
  input_df = pd.DataFrame({'x': x_trajectory[10:-1], 'y': y_trajectory[10:-1], 'z': z_trajectory[10:-1]})
  output_df = pd.DataFrame({'x': x_trajectory[11:], 'y': y_trajectory[11:], 'z': z_trajectory[11:]})
  return input_df, output_df

# Training data

# Set the initial conditions [x0, y0, z0]
initial_conditions = [1.0, 0.0, 0.0]

# Define the time points for propagation
t = np.linspace(0, 1000, 10000)  # 1000 seconds with 10000 time points

input_df, output_df = dataset_gen(initial_conditions, t)

# Save the DataFrame to a CSV file
input_df.to_csv('lorenz_train_inputs.csv', index=False)
output_df.to_csv('lorenz_train_outputs.csv', index=False)

# Test data
# Set the initial conditions [x0, y0, z0]
test_initial_conditions = [10, 20.0, 5.0]

# Define the time points for propagation
t = np.linspace(0, 1000, 10000)  # 1000 seconds with 10000 time points

test_input_df, test_output_df = dataset_gen(test_initial_conditions, t)

# Save the DataFrame to a CSV file
test_input_df.to_csv('lorenz_test_inputs.csv', index=False)
test_output_df.to_csv('lorenz_test_outputs.csv', index=False)