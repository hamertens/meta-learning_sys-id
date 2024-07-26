import os
import pandas as pd

class Dataset:
    def __init__(self, data_dir_name):
        self.data_dir = os.path.join(os.path.dirname(__file__), data_dir_name)

    def load_data(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        return pd.read_csv(file_path).values

    def load_sample_inputs(self):
        return self.load_data('sample_inputs.csv')

    def load_sample_outputs(self):
        return self.load_data('sample_outputs.csv')

    def load_test_inputs(self):
        return self.load_data('test_inputs.csv')

    def load_test_outputs(self):
        return self.load_data('test_outputs.csv')
    
    def load_validation_inputs(self):
        return self.load_data('validation_inputs.csv')
    
    def load_validation_outputs(self):
        return self.load_data('validation_outputs.csv')

class Lorenz(Dataset):
    def __init__(self):
        self.dataframe_columns_input = ['x', 'y', 'z']
        self.dataframe_columns_output = ['x', 'y', 'z']
        super().__init__('lorenz')

class Pendulum(Dataset):
    def __init__(self):
        self.dataframe_columns_input = ['theta', 'omega']
        self.dataframe_columns_output = ['theta', 'omega']
        super().__init__('pendulum')

class DoublePendulum(Dataset):
    def __init__(self):
        self.dataframe_columns_input = ["theta1", "theta2", "omega1", "omega2"]
        self.dataframe_columns_output = ["theta1", "theta2", "omega1", "omega2"]
        super().__init__('double_pendulum')

class TwoTankSystem(Dataset):
    def __init__(self):
        self.dataframe_columns_input = ['h1', 'h2', "q"]
        self.dataframe_columns_output = ['h1', 'h2']
        super().__init__('two_tank_system')

class ActuatedPendulum(Dataset):
    def __init__(self):
        self.dataframe_columns_input = ['theta', 'omega', 'torque']
        self.dataframe_columns_output = ['theta', 'omega']
        super().__init__('actuated_pendulum')



