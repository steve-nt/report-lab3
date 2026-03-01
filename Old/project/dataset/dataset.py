# Import configuration classes for dataset and plotting
from config import ConfigDataset, ConfigPlot
# Import specific dataset classes
from dataset.download.b_alzheimer5100 import Balzheimer5100
from dataset.download.b_tumor3000 import Btumor3000
from dataset.download.b_tumor4600 import Btumor4600
# Import the DatasetGenerator for processing raw datasets
from dataset.generator import DatasetGenerator

# Define the Dataset class for managing multiple datasets
class Dataset:
    """
    Packages and manages all downloaded datasets together.
    Provides methods to merge datasets or retrieve specific subsets for training.
    """
    # List to store all datasets with their metadata
    datasets = []
    # Store dataset configuration for consistent preprocessing
    dataset_config = None
    # Store plot configuration for visualization
    plot_config = None
    
    def __init__(self, datasets : list, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        """
        Initialize the Dataset manager with datasets and configurations.

        Parameters
        ----------
            datasets : list[(str(ID), dataset, [int,int])]
                List of tuples containing:
                - ID (str): unique identifier for the dataset
                - dataset: dataset object with paths and labels
                - ranges: list of index ranges to subset the data (empty list = use all)
            dataset_config : ConfigDataset 
                Configuration for image resizing and normalization
            plot_config : ConfigPlot 
                Configuration for visualizing dataset samples
        """
        # Store the list of datasets
        self.datasets = datasets
        # Store the dataset configuration
        self.dataset_config = dataset_config
        # Store the plot configuration
        self.plot_config = plot_config
    
    def split_indicies(self, result_paths, result_labels, indicies):
        """
        Extract a subset of data based on specified index ranges.
        
        Parameters:
            result_paths: List of file paths
            result_labels: Corresponding labels for the paths
            indicies: List of [start, end] ranges to extract
            
        Returns:
            Subset of paths and labels within the specified ranges
        """
        # Validate index range is properly ordered
        if(0 <= indicies[0] < indicies[1]):
            # Return sliced paths and labels within the specified range
            return result_paths[indicies[0]:indicies[1]], result_labels[indicies[0]:indicies[1]]
        else:
            # Raise exception if index range is invalid
            raise Exception("Range in dataset is incorrect")
    
    def mergeAll(self):
        """
        Merge all datasets into a single dataset for training.
        
        Returns
        ----------
        tuple : (train_data, validation_data, test_data)
            TensorFlow iterators for training, validation, and test datasets
            
        Raises
        ----------
        Exception
            If no datasets are loaded or dataset tuple structure is invalid
        """
        
        # Check that at least one dataset is provided
        if len(self.datasets) == 0:
            raise Exception("No datasets loaded in DatasetGenerator.")
        # Validate that each dataset tuple has exactly 3 elements
        for x in self.datasets:
            if len(x) != 3:
                raise Exception("Must specify name (string), dataset (ImportKaggle), indicies of dataset ([int,int])")
        
        # Initialize lists to collect paths and labels from all datasets
        result_paths = []
        result_labels = []
        result_indicies = []
        
        # Debug print showing all datasets being merged
        print(self.datasets)
        # Iterate through each dataset tuple
        for x in self.datasets:
            print(x)
            # Collect all file paths from this dataset
            for y in x[1].paths:
                result_paths.append(y)
            # Collect all labels from this dataset
            for y in x[1].labels:
                result_labels.append(y)
            # Collect index ranges for subsetting
            result_indicies = result_indicies + x[2]
                
        # Create a DatasetGenerator instance to process the merged data
        dataset = DatasetGenerator()
        
        # Generate and return train, validation, and test data iterators
        return dataset.generate(result_paths, result_labels, result_indicies, self.dataset_config, self.plot_config, None)
        
    def get(self, *args):
        """
        Retrieve a specific dataset or merged subset of datasets by index or indices.
        
        Returns
        ----------
        tuple : (train_data, validation_data, test_data)
            TensorFlow iterators for the requested dataset(s)
            
        Raises
        ----------
        Exception
            If dataset indices are out of range
        """
        # Initialize lists to collect paths, labels, and index ranges
        result_paths = []
        result_labels = []
        result_indicies = []
        
        # Handle single dataset index (int argument)
        if len(args) == 1 and isinstance(args[0], int):
            # Get the dataset index
            i = args[0]
            # Validate that the index is within range
            if(0 <= i <= (len(self.datasets) - 1)):
                # Extract paths and labels from the specified dataset
                result_paths = self.datasets[i][1].paths
                result_labels = self.datasets[i][1].labels
                # Add index ranges for this dataset
                result_indicies = result_indicies + self.datasets[i][2]
                
                # Create dataset generator and process the data
                dataset = DatasetGenerator()
                
                # Generate and return train, val, test iterators with dataset-specific preprocessing
                return dataset.generate(result_paths, result_labels, result_indicies, self.dataset_config, self.plot_config, lambda image: self.datasets[i][1].pre_processing(image))
            else:
                raise Exception("Index out of range.")
        
        # Handle multiple dataset indices (list argument)
        if len(args) == 1 and isinstance(args[0], list):
            # Iterate through each requested dataset index
            for i in args[0]:
                # Validate that each index is within range
                if(0 <= i <= (len(self.datasets) - 1)):
                    # Append paths from this dataset to the result
                    result_paths = result_paths + self.datasets[i][1].paths
                    # Append labels from this dataset to the result
                    result_labels = result_labels + self.datasets[i][1].labels
                    # Append index ranges from this dataset
                    result_indicies = result_indicies + self.datasets[i][2]
                else:
                    raise Exception("Index out of range.")
                
            # Create dataset generator for the merged datasets
            dataset = DatasetGenerator()
            # Generate and return train, val, test iterators (uses last dataset's preprocessing)
            return dataset.generate(result_paths, result_labels, result_indicies, self.dataset_config, self.plot_config, lambda image: self.datasets[i][1].pre_processing(image))
        
    def print(self):
        """
        Debug utility to print all loaded datasets and their configurations.
        """
        # Iterate and print each dataset tuple
        for x in (self.datasets):
            print(x)
  


        