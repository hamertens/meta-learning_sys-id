import numpy as np
import os
#from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
os.environ['DDE_BACKEND'] = 'tensorflow'
import deepxde as dde



class MCDropout:
    def __init__(self, training_type, sample_inputs, sample_outputs, test_inputs, test_outputs, training_inputs, training_outputs, hps):
        self.training_type = training_type
        self.output_data_dir = os.path.join(os.path.dirname(__file__), '..', 'output_data')
        self.learning_rate = hps['lr']
        self.forward_passes = hps['forward_passes']
        self.num_neurons = hps['num_neurons']
        self.num_layers = hps['num_layers']
        self.epochs = hps['epochs']
        self.batch_size = hps['batch_size']
        self.input_dim = sample_inputs.shape[1]
        self.output_dim = sample_outputs.shape[1]

        self.sample_inputs = sample_inputs
        self.sample_outputs = sample_outputs
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        self.test_inputs = test_inputs
        self.test_outputs = test_outputs
    
    # Define the BNN
    def train_dropout_model(self, init_script=False):

        layer_size = [self.input_dim] + [self.num_neurons] * self.num_layers + [self.output_dim]
        activation = "sigmoid"
        initializer = "Glorot uniform"
        regularization = ["l2", 1e-5]
        dropout_rate = 0.01
        net = dde.nn.FNN(
            layer_size,
            activation,
            initializer,
            regularization,
            dropout_rate
        )
        data = dde.data.DataSet(X_train=self.training_inputs, y_train=self.training_outputs, X_test=self.sample_inputs, y_test=self.sample_outputs)
        BNN_model = dde.Model(data, net)
        BNN_uncertainty = dde.callbacks.DropoutUncertainty(period=self.forward_passes)
        BNN_model.compile("adam", lr=self.learning_rate, metrics=["l2 relative error"])
        if self.training_type == "continuous":
            save_path = 'output_data/model/'
            if init_script == True:
                load_path = None
            else:
                load_path = 'output_data/model/-' + str(self.epochs) + '.weights.h5'
        else:
            save_path = None
            load_path = None   
        _, train_state = BNN_model.train(iterations=self.epochs, callbacks= [BNN_uncertainty], model_save_path=save_path 
                                                        ,model_restore_path=load_path)
        del data
        del net
        return train_state, BNN_model
        
    
    def first_run(self):
        _, model = self.train_dropout_model(init_script=True)
        Y_pred = model.predict(self.training_inputs)
        return Y_pred
    
    def active_learning(self):
        train_state, model = self.train_dropout_model(init_script=False)
        # Find the sample with the highest uncertainty
        Y_std = train_state.y_std_test
        sum_values = Y_std.sum(axis=1)
        max_index = np.argmax(sum_values)
        # Calculate RMSE
        prediction = model.predict(self.test_inputs)
        RMSE = np.sqrt(mean_squared_error(self.test_outputs, prediction))
        # Calculate new prediction and total variance
        Y_pred = train_state.y_pred_test
        new_prediction = Y_pred[max_index]
        variance = np.sum(Y_std)
        return max_index, new_prediction, variance, RMSE
        