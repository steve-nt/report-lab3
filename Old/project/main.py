# Import standard library modules for file path handling
import os
# Import TensorFlow for deep learning operations
import tensorflow as tf
# Import random for shuffling and random selection
import random
# Import NumPy for numerical operations
import numpy as np

# Import the Dataset class for handling and managing datasets
from dataset.dataset import Dataset
# Import configuration classes for different components
from config import ConfigFederated, ConfigOod, ConfigModel, ConfigDataset, ConfigPlot
# Import dataset download classes for specific datasets
from dataset.download.a_faces16000 import Afaces16000
from dataset.download.b_alzheimer5100 import Balzheimer5100
from dataset.download.b_alzheimer5100_poisoned import Balzheimer5100_poisoned
from dataset.download.b_tumor3000 import Btumor3000
from dataset.download.b_tumor4600 import Btumor4600
from dataset.download.l_pneumonia5200 import Lpneumonia5200
# Import the Federated learning class for running federated simulations
from federated.federated import Federated
# Import the Model class for building and training neural networks
from model.model import Model

# Import plotting utility for model visualization
from model.math.plot import ModelPlot

#
# FEEL FREE TO EDIT THE CONTENT OF ALL GIVEN FILES AS YOU LIKE.
#

############# REPRODUCIBILITY, deterministic behavior #############
def set_seeds(SEED):
    """ Set seeds for deterministic behavior across all libraries.
    This ensures reproducible results across runs by controlling randomness in Python, TensorFlow, and NumPy.
    """
    # Set Python's hash seed for deterministic dictionary/set ordering
    os.environ['PYTHONHASHSEED'] = str(SEED)
    # Set Python's random module seed
    random.seed(SEED)
    # Set TensorFlow's random seed
    tf.random.set_seed(SEED)
    # Set NumPy's random seed
    np.random.seed(SEED)

def set_global_determinism(SEED):
    """
    Configure global determinism settings across all libraries.
    This function calls set_seeds and optionally enables additional determinism flags.
    """
    # Call set_seeds to initialize all random number generators with the same seed
    set_seeds(SEED=SEED)
    
    # Uncomment below if limiting cpu treads, may help with determinism. However may slow down training, especially on large models.
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # CUDA/GPU users - Uncomment below for GPU determinism (may impact performance)
    # os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # tf.config.experimental.enable_op_determinism()

# Set the random seed value for reproducibility (change as desired)
SEED = 42   
# Initialize global determinism with the selected seed
set_global_determinism(SEED=SEED)
###############################################################

class ModelSimulation():
    """ Only for testing single model. 
    """
           
    def run(self):
        #-----------CONFIG--------------------
        model_config = ConfigModel(
            debug = True,
            epochs = 5,
            activation = 'relu',
            activation_out = 'softmax',
            optimizer = 'adam',
            loss = 'categorical_crossentropy'
        )
        dataset_config = ConfigDataset(
            debug = True,                   # DISABLE IF YOU WANT TO PREVENT IMAGE EXAMPLES FROM BEING DISPLAYED BEFORE TRAINING.
            batch_size = 64,                          
            image_size = 256,            
            input_shape = (256,256,1),   
            split = 0.25,
            number_of_classes=2
        )
        plot_config = ConfigPlot(
            plot = True,
            path = "./.env/.saved/",
            img_per_class = 10  
        )
        
        #-----------SIM---------------------- 
        m = Model(
            model_config=model_config,
            dataset_config=dataset_config,
            plot_config=plot_config
        )
        
        # Create dataset by merging multiple datasets together. 
        train_data, validation_data, test_data = Dataset(
            [
                (Btumor4600().ID, Btumor4600(), []),
                (Btumor3000().ID, Btumor3000(), []),
                (Balzheimer5100().ID, Balzheimer5100(), []),
                (Lpneumonia5200().ID, Lpneumonia5200(), [])
                
            ],
            dataset_config=dataset_config,
            plot_config=plot_config
        ).mergeAll()
        
        # Below is an example of dataset with subsets. That can be used in federated learning context. 
        # All given datasets down below are available in ./dataset/download.
        #
        # dataset = Dataset(
        #     [
        #         (Btumor4600().ID, Btumor4600(), []),    # id 0 ID DATA 
        #         (Btumor3000().ID, Btumor3000(), []),    # id 1 ID DATA 
        #         (Balzheimer5100().ID, Balzheimer5100(), []), # id 2 ID DATA 
        #         (Lpneumonia5200().ID, Lpneumonia5200(), []), # id 3 ID DATA 
        #         (Lpneumonia5200().ID, Lpneumonia5200(), [[300,500],[3600,3800]]), # id 4 ID DATA (subsamples of total, not used in pre-training) 
        #         (Btumor4600().ID, Btumor4600(), [[300,500],[3700,3900]]), # id 5 ID DATA (subsamples of total, not used in pre-training)  
        #         (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), [[1000,1700],[4000,4700]]), # id 6 OOD DATA (poisoned data) (not used in pre-training)
        #         (Afaces16000().ID, Afaces16000(), [[1,700],[2501,3200]]) # id 7 OOD DATA (not used in pre-training) # Take some two subsets of complete dataset. # [250,750], [4000,4500]
        #     ],
        #     dataset_config=dataset_config,
        #     plot_config=plot_config
        # )
        #
        # Bind local model / client to dataset id / subsets.
        #
        # train_data, validation_data, test_data = dataset.get(index i)
        #
        
        m.train(train_data, validation_data)     
        m.test(test_data) 
        
        m.plot_all(test_data, xlabel="CNN Model", title="CNN Model")
        input('Press Enter to close plots and exit...')
        
class FederatedSimulation():
    """
        TODO:
        Simulation for federated learning with multiple local models.
        You are free to change the configuration as you like, remove or edit.
        
        This are only some parameters for simulation that might be of interest.
        
        You may remove the configuration entirely if you prefer and apply an alternative setup.
    """
    #--------------------------------------- 
    #--------------CONFIG-------------------
    #--------------------------------------- 
    federated_config = ConfigFederated(
        debug=True,
        save=True,
        load=False,
        load_round=0,
        load_reg=True,
        path="./.env/.saved/",
        rounds=2,          
        clients=5,
        participants=4,
        host_id=0,
        client_to_dataset=[[0,1,2,3],[0],[1],[2],[3]],
        ood_round=26,
        delete_on_load=False,
)
    
    ood_config = ConfigOod(
        debug = True,
        hdc_debug = False,
            
        # _______SIMULATION________
        enabled = False,                            # Enabling hdff and ood. 
        hyper_size=int(1e4),                        # Hyper dimensions for projection matrix. 
        
        id_client = [1,2,3,4],                      # Clients id in-distribution, excluding global client/model.
        ood_client = [5,6],                           # Datasets out-of-distribution (index in dataset), excluding global client/model.
        
        ood_protection = True,                     # IF ood protection (exluding) is enabled.
        ood_protection_thres = 0.7                  # Threshold for consider models being ood. 
    )
    
    model_config = ConfigModel(
        debug = True,
            
        # _______SIMULATION________
        epochs = 1,                         # Highly recommend to be 1 for federated. Change rounds instead. 
        activation = 'relu',                # Parameters for model.
        activation_out = 'softmax',
        optimizer = 'adam',
        loss = 'categorical_crossentropy'   # 'categorical_crossentropy'
    )
    dataset_config = ConfigDataset(
        debug = False,
            
        # _______SIMULATION________
        batch_size = 64,                    # Batchsize for datasets.                          
        image_size = 256,                   # Image size 
        input_shape = (256,256,1),
        split = 0.25,                       # Train / validation split.
        number_of_classes = 2               # Number of total classes.
    )
    plot_config = ConfigPlot(
        plot = False,
            
        # __________FILE___________
        path = './.env/plot',                
        img_per_class = 10                  # Only for plotting example pictures from dataset (debug must be set to true).
    )
    
    def run(self):
        #------------------------------------ 
        #--------------SIM------------------- 
        #------------------------------------ 
        m = Model(
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        # client-to-dataset and id dataset and ood dataset are referenced to this list. 
        dataset = Dataset(
            [
                (Btumor4600().ID, Btumor4600(), []),    # id 0 ID DATA 
                (Btumor3000().ID, Btumor3000(), []),    # id 1 ID DATA 
                (Balzheimer5100().ID, Balzheimer5100(), []), # id 2 ID DATA 
                (Lpneumonia5200().ID, Lpneumonia5200(), []), # id 3 ID DATA 
                (Lpneumonia5200().ID, Lpneumonia5200(), [[300,500],[3600,3800]]), # id 4 ID DATA (subsamples of total, not used in pre-training) 
                (Btumor4600().ID, Btumor4600(), [[300,500],[3700,3900]]), # id 5 ID DATA (subsamples of total, not used in pre-training) 
                (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), [[1000,1700],[4000,4700]]), # id 6 OOD DATA (poisoned data) (not used in pre-training)
                (Afaces16000().ID, Afaces16000(), [[1,700],[2501,3200]]) # id 7 OOD DATA (not used in pre-training) # Take some two subsets of complete dataset. # [250,750], [4000,4500]
            ],
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        federated = Federated(
            dataset=dataset, 
            model=m,
            federated_config=self.federated_config,
            ood_config=self.ood_config, 
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        return federated.run()

class Task3Simulation(FederatedSimulation):
    """FederatedSimulation with OOD detection enabled. Overrides only the relevant configs."""
    federated_config = ConfigFederated(
        **{**vars(FederatedSimulation.federated_config),
           "rounds": 3,
           "ood_round": 2,
           "clients": 6,
           "client_to_dataset": [[0,1,2,3],[0],[1],[2],[3],[6]]}
    )
    ood_config = ConfigOod(
        **{**vars(FederatedSimulation.ood_config),
           "enabled": True,
           "id_client": [1,2,3,4],
           "ood_client": [5]}
    )

if __name__ == "__main__":
    TASK = 2  # Change to 1, 2, or 3

    if TASK == 1:
        ModelSimulation().run()
    elif TASK == 2:
        FederatedSimulation().run()
    elif TASK == 3:
        Task3Simulation().run()
