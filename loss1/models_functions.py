import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input
#from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from keras_uncertainty.losses import regression_gaussian_nll_loss

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from setup import INPUT_DIMENSIONALITY, OUTPUT_DIMENSIONALITY


# Read DataFrame from CSV file
hyperparameters_df = pd.read_csv('output_data/hyperparameters.csv')

# Extract the values from the DataFrame
learning_rate = hyperparameters_df['lr'][0]
num_neurons = 120
num_layers = 1

def mlp_model():
    inp = Input(shape=(INPUT_DIMENSIONALITY,))
    x = inp
    for _ in range(num_layers):
        x = Dense(num_neurons, activation="sigmoid")(x)
    mean = Dense(OUTPUT_DIMENSIONALITY, activation="linear")(x)
    var = Dense(OUTPUT_DIMENSIONALITY, activation="softplus")(x)

    train_model = Model(inp, mean)
    pred_model = Model(inp, [mean, var])

    # Create an instance of the Adam optimizer with the desired learning rate
    optimizer = {"class_name": "adam", "config": {"learning_rate": learning_rate}}
    train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer=optimizer)

    return train_model, pred_model




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