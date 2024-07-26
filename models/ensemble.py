import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
#from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from keras_uncertainty.losses import regression_gaussian_nll_loss
from sklearn.metrics import mean_squared_error
from keras_uncertainty.models import DeepEnsembleRegressor
from keras.callbacks import EarlyStopping
import os
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class Ensemble:
    def __init__(self, training_type, sample_inputs, sample_outputs, test_inputs, test_outputs, validation_inputs, validation_outputs, training_inputs, training_outputs, hps, output_folder):
        self.training_type = training_type
        self.output_data_dir = os.path.join(os.path.dirname(__file__), '..', output_folder)
        self.learning_rate = hps['lr']
        self.num_neurons = hps['num_neurons']
        self.num_layers = hps['num_layers']
        self.num_models = hps['num_models']
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
        self.validation_inputs = validation_inputs
        self.validation_outputs = validation_outputs
    
    def mlp_model(self):
        inp = Input(shape=(self.input_dim,))
        x = inp
        for _ in range(self.num_layers):
            x = Dense(self.num_neurons, activation="sigmoid")(x)
        mean = Dense(self.output_dim, activation="linear")(x)
        var = Dense(self.output_dim, activation="softplus")(x)

        train_model = Model(inp, mean)
        pred_model = Model(inp, [mean, var])

        # Create an instance of the Adam optimizer with the desired learning rate
        optimizer = {"class_name": "adam", "config": {"learning_rate": self.learning_rate}}
        train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer=optimizer)

        return train_model, pred_model
        

    def find_max_uncertainty(self):
            Y_pred, Y_std = self.model.predict(self.sample_inputs)
            sum_values = Y_std.sum(axis=1)
            max_index = np.argmax(sum_values)
            return max_index, Y_pred[max_index], np.sum(Y_std)
    
    def first_run(self):
        self.model = DeepEnsembleRegressor(self.mlp_model, self.num_models)
        self.model.fit(self.training_inputs, self.training_outputs, epochs=self.epochs, batch_size = self.batch_size)
        if self.training_type == "continuous":
            self.model.save_weights(os.path.join(self.output_data_dir, 'ensemble_model_weights'))
            self.model.save_weights_training(os.path.join(self.output_data_dir, 'ensemble_model_weights_training'))

        Y_pred, _ = self.model.predict(self.training_inputs)
        return Y_pred
    
    def active_learning(self):
        early_stopping = EarlyStopping(
            monitor='val_loss',    # Monitor validation loss
            patience=3,            # Number of epochs to wait for improvement
            min_delta=0.01,       # Minimum change to qualify as an improvement
            restore_best_weights=True  # Restore model weights from the best epoch
        )
        self.model = DeepEnsembleRegressor(self.mlp_model, self.num_models)
        if self.training_type == "continuous":
            self.model.load_weights(os.path.join(self.output_data_dir, 'ensemble_model_weights/'))
            self.model.load_weights_training(os.path.join(self.output_data_dir, 'ensemble_model_weights_training/'))

            for test_model, train_model in zip(self.model.test_estimators, self.model.train_estimators):
                last_layer_name = test_model.layers[-1].name
                optimizer = {"class_name": "adam", "config": {"learning_rate": self.learning_rate}}
                train_model.compile(loss=regression_gaussian_nll_loss(test_model.get_layer(last_layer_name).output), optimizer=optimizer)


        self.model.fit(self.training_inputs, self.training_outputs, epochs=self.epochs, batch_size = self.batch_size, 
                       validation_data=(self.validation_inputs, self.validation_outputs), callbacks=[early_stopping])
        max_index, new_prediction, variance = self.find_max_uncertainty()
        #Append RMSE and variance
        prediction, _ = self.model.predict(self.test_inputs)
        # Append the new array to training inputs
        RMSE = np.sqrt(mean_squared_error(self.test_outputs, prediction))

        return max_index, new_prediction, variance, RMSE
        