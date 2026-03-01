import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten

from config import ConfigDataset, ConfigModel, ConfigPlot
from model.math.plot import ModelPlot

class Model:
    """
    CNN Model wrapper for Lab 3: Federated Learning + OOD Detection.
    
    Implements a Convolutional Neural Network (CNN) that is:
    1. Instantiated once per client (global model + local models) in the FL environment
    2. Trained locally on each client's assigned dataset
    3. Aggregated via Federated Averaging (FedAvg) to update the global model
    4. Used for OOD detection via HDFF feature extraction and comparison
    
    The CNN architecture has Conv2D layers with MaxPooling for feature extraction,
    followed by Dense layers for classification (binary: sick/healthy).
    Related to section 2.1.1 and Task 2 (Phase 1 training).
    """
    def __init__(self, model_config : ConfigModel, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        """
        Single model structure for training / classification. 

        Parameter
        ----------
            model_config : ConfigModel
                Model config, properties for neural network.
            dataset_config : ConfigDataset
                Dataset config, how input shape in model should be.
            plot_config : ConfigPlot
                Plotting config, plotting history and confusion matrix.
        """
        self.plot = ModelPlot()
        self.model = None
        
        self.epochs = 0         # How many epochs was runned for model (most useful for federated env. to see how much local model contributed to global model).
        
        self.acc = None         # For training / validation plotting.
        self.val_acc = None
        self.loss = None
        self.val_loss = None

        self.test_accuracy = [] # For test plotting. 
        self.test_loss = []
        
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.plot_config = plot_config
        
        # ------------------------------- MODEL (CNN Architecture) -------------------------------
        # Builds a CNN with 4 Conv2D blocks followed by Dense layers for binary classification.
        # This model is used for all clients (global + local) in the FL simulation.
        
        model = Sequential()
            
        model.add(InputLayer(input_shape=self.dataset_config.input_shape, batch_size=self.dataset_config.batch_size))
        model.add(Conv2D(16, 3, activation=self.model_config.activation))
        model.add(MaxPooling2D())

        model.add(Conv2D(32, 3, activation=self.model_config.activation))
        model.add(MaxPooling2D())

        model.add(Conv2D(32, 3, activation=self.model_config.activation))
        model.add(MaxPooling2D())

        model.add(Conv2D(16, 3, activation=self.model_config.activation))
        model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(256, activation=self.model_config.activation))
        model.add(Dense(2, activation=self.model_config.activation_out))
        
        # Compile with specified optimizer, loss, and metrics
        model.compile(optimizer=self.model_config.optimizer, loss=self.model_config.loss, metrics=['accuracy'])
        
        if(self.model_config.debug):
            model.summary()
        
        self.model = model

    def train(self, train_data, val_data):
        """
        Trains model and evaluate with test data. 
        
        Part of Task 2.3 (Local Training): Each local client trains on its assigned data
        for the configured number of epochs. Training history is accumulated over multiple
        federated rounds.

        Parameters
        ----------
            train_data : tensorflow.data.Dataset
                train data for training.
            val_data : tensorflow.data.Dataset
                validation data for training.
        """
        
        history = self.model.fit(train_data,
                    epochs=int(self.epochs + self.model_config.epochs),
                    validation_data=val_data,
                    initial_epoch=self.epochs)
        
        if(self.epochs == 0):
            self.acc = history.history['accuracy']
            self.val_acc = history.history['val_accuracy']

            self.loss = history.history['loss']
            self.val_loss = history.history['val_loss']
        else:
            self.acc += history.history['accuracy']
            self.val_acc += history.history['val_accuracy']

            self.loss += history.history['loss']
            self.val_loss += history.history['val_loss']
            
        self.epochs = int(self.epochs + self.model_config.epochs)
        
    def test(self, test_data):
        """
        Evaluates model on test data.
        
        Part of Task 2.5 (Global Evaluation): The global model is tested on all
        in-distribution test data to track convergence over federated rounds.

        Parameters
        ----------
            test_data : tensorflow.data.Dataset
                Test data for evaluation.
        """
        loss, accuracy = self.model.evaluate(test_data) # Evaluate model on test data.
        
        self.test_accuracy.append(accuracy)
        self.test_loss.append(loss)

        if(self.model_config.debug):
            print("Test loss= ", loss)
            print("Test accuracy=", accuracy)
            print()
            
    def plot_all(self, test_data, xlabel : str, title : str):
        """Plot training history and confusion matrix."""
        if(self.acc != None):
            self.plot.accuracy(self.acc, self.val_acc, self.loss, self.val_loss, xlabel, title)
        self.plot.confusion_matrix(self.model, test_data, title)
        
    def plot_test(self, xlabel : str, title : str):
        """Plot test accuracy and loss over time."""
        self.plot.test_accuracy(self.test_accuracy, self.test_loss, xlabel, title)
    