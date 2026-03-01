# Import TensorFlow library for deep learning operations
import tensorflow as tf
# Import Sequential model from Keras for creating sequential neural networks
from tensorflow.keras.models import Sequential
# Import layers that will be used in the neural network architecture
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten

# Import configuration classes for model, dataset, and plotting settings
from config import ConfigDataset, ConfigModel, ConfigPlot
# Import plotting utility for model performance visualization
from model.math.plot import ModelPlot

# Define the Model class for training and testing neural networks
class Model:
    def __init__(self, model_config : ConfigModel, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        """
        Initialize the Model class with configuration and build CNN architecture.

        Parameter
        ----------
            model_config : ConfigModel
                Model configuration specifying neural network hyperparameters (epochs, activation, optimizer, loss).
            dataset_config : ConfigDataset
                Dataset configuration specifying input shape and batch size for the model.
            plot_config : ConfigPlot
                Plotting configuration for visualizing model performance and confusion matrix.
        """
        # Initialize the plotting utility for generating visualization plots
        self.plot = ModelPlot()
        # Store the actual Keras sequential model (will be created below)
        self.model = None
        
        # Track total epochs run (useful in federated learning to see client contributions)
        self.epochs = 0         
        
        # Lists to store training history metrics for plotting (training metrics)
        self.acc = None         # Training accuracy history
        self.val_acc = None     # Validation accuracy history
        self.loss = None        # Training loss history
        self.val_loss = None    # Validation loss history

        # Lists to store test performance metrics across all test evaluations
        self.test_accuracy = [] # Test accuracy for each evaluation
        self.test_loss = []     # Test loss for each evaluation
        
        # Store configuration objects for reference during training
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.plot_config = plot_config
        
        # ------------------------------- MODEL ARCHITECTURE -------------------------------
        
        # Create a sequential model (layers stacked sequentially)
        model = Sequential()
            
        # Add input layer that specifies the shape of input data (batch_size, height, width, channels)
        model.add(InputLayer(input_shape=self.dataset_config.input_shape, batch_size=self.dataset_config.batch_size))
        # Add first convolutional layer with 16 filters, 16x16 kernel, and specified activation function
        model.add(Conv2D(16, 16, activation=self.model_config.activation))
        # Add max pooling layer to reduce spatial dimensions and extract prominent features
        model.add(MaxPooling2D())
            
        # Add second convolutional layer with 16 filters and max pooling
        model.add(Conv2D(16, 16, activation=self.model_config.activation))
        model.add(MaxPooling2D())
        
        # Add third convolutional layer with 16 filters and max pooling
        model.add(Conv2D(16, 16, activation=self.model_config.activation))
        model.add(MaxPooling2D())
            
        # Add fourth convolutional layer with 16 filters and max pooling
        model.add(Conv2D(16, 16, activation=self.model_config.activation))
        model.add(MaxPooling2D())
            
        # Flatten the 2D spatial output into a 1D vector for dense layers
        model.add(Flatten())
        # Add fully connected (dense) layer with 32 neurons and specified activation
        model.add(Dense(32, activation=self.model_config.activation))        
        # Add output layer with 2 neurons (for binary classification: sick/healthy) and softmax activation
        model.add(Dense(2, activation=self.model_config.activation_out))
        
        # ---------------------------------------------------------------------
        
        # Compile the model with specified optimizer, loss function, and metrics
        model.compile(optimizer=self.model_config.optimizer, loss=self.model_config.loss, metrics=['accuracy'])
        
        # If debug mode is enabled, print the model architecture summary
        if(self.model_config.debug):
            model.summary()
        
        # Store the compiled model as an instance variable
        self.model = model

    def train(self, train_data, val_data):
        """
        Train the model on training data and evaluate on validation data. 

        Parameters
        ----------
            train_data : TensorFlow Dataset
                Training data (images and labels) for model training.
            val_data : TensorFlow Dataset
                Validation data for evaluating model during training.
        """
        
        # Train the model with specified number of epochs, starting from where it left off
        history = self.model.fit(train_data,
                    # Calculate total epochs: current epochs already run + new epochs to run
                    epochs=int(self.epochs + self.model_config.epochs),
                    # Provide validation data to evaluate model after each epoch
                    validation_data=val_data,
                    # Continue training from the last epoch (for federated learning continuity)
                    initial_epoch=self.epochs)
        
        # If this is the first training session, initialize the history lists
        if(self.epochs == 0):
            # Store training accuracy values from this training session
            self.acc = history.history['accuracy']
            # Store validation accuracy values from this training session
            self.val_acc = history.history['val_accuracy']

            # Store training loss values from this training session
            self.loss = history.history['loss']
            # Store validation loss values from this training session
            self.val_loss = history.history['val_loss']
        else:
            # If continuing training, append new history to existing history (for tracking across rounds)
            self.acc += history.history['accuracy']
            self.val_acc += history.history['val_accuracy']

            # Append training and validation loss to existing history
            self.loss += history.history['loss']
            self.val_loss += history.history['val_loss']
            
        # Update total epochs run (used to track model training progress in federated learning)
        self.epochs = int(self.epochs + self.model_config.epochs)
        
        
    def test(self, test_data):
        """
        Evaluate model on test data and store results.
        
        Parameters
        ----------
            test_data : TensorFlow Dataset
                Test dataset containing images and labels for evaluation.
        """
        # Run model evaluation on test data and retrieve loss and accuracy metrics
        loss, accuracy = self.model.evaluate(test_data)
        
        # Store test accuracy value in list for tracking across multiple test evaluations
        self.test_accuracy.append(accuracy)
        # Store test loss value in list for tracking across multiple test evaluations
        self.test_loss.append(loss)

        # If debug mode is enabled, print the test results to console
        if(self.model_config.debug):
            print("Test loss= ", loss)
            print("Test accuracy=", accuracy)
            print()
            
            
    def plot_all(self, test_data, xlabel : str, title : str):
        """
        Generate and display all plots: accuracy, loss, and confusion matrix.
        
        Parameters
        ----------
            test_data : TensorFlow Dataset
                Test data used for generating confusion matrix.
            xlabel : str
                Label for x-axis in plots.
            title : str
                Title for all plots.
        """
        # If training history exists (model was trained), plot accuracy and loss curves
        if(self.acc != None):
            self.plot.accuracy(self.acc, self.val_acc, self.loss, self.val_loss, xlabel, title)
        # Generate and display confusion matrix showing prediction vs actual labels
        self.plot.confusion_matrix(self.model, test_data, title)
        
    def plot_test(self, xlabel : str, title : str):
        """
        Plot test accuracy and loss across multiple test evaluations.
        
        Parameters
        ----------
            xlabel : str
                Label for x-axis (typically round or iteration number).
            title : str
                Title for the plot.
        """
        # Generate plot showing test accuracy and loss over multiple evaluations
        self.plot.test_accuracy(self.test_accuracy, self.test_loss, xlabel, title)
    