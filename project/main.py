# Lab 3: Out-of-Distribution Detection in Federated Learning using Hyperdimensional Computing
# This file implements the main entry point and simulation classes for the lab.
#
# Structure:
# - configure_metal(): GPU acceleration for Apple Silicon (M4)
# - set_seeds(): Reproducibility setup
# - ModelSimulation: Single model training (Task 1)
# - FederatedSimulation: Phase 1 pre-training without OOD detection (Task 2)
# - Task3Simulation: Phase 2 with OOD detection enabled (Task 3)
# - Task4Base + Task4PreTrain + Task4Exp[1-5]: Task 4 experiments (Phase 2 with various OOD scenarios)
#
# Assignment references:
# - Lab objective: Section 1-2
# - Phase 1: Section 2.2.2 & 3.2
# - Phase 2 + OOD: Section 2.2.2 & 3.3-3.4
# - Experiments: Section 3.4

# Non-interactive backend: plots are saved to disk, plt.show() is a no-op.
# Comment out these 2 lines if you want interactive plot windows (Tasks 1-3).
import matplotlib
matplotlib.use('Agg')

import os
import tensorflow as tf
import random
import numpy as np

from dataset.dataset import Dataset
from config import ConfigFederated, ConfigOod, ConfigModel, ConfigDataset, ConfigPlot
from dataset.download.a_faces16000 import Afaces16000
from dataset.download.b_alzheimer5100 import Balzheimer5100
from dataset.download.b_alzheimer5100_poisoned import Balzheimer5100_poisoned
from dataset.download.b_tumor3000 import Btumor3000
from dataset.download.b_tumor4600 import Btumor4600
from dataset.download.l_pneumonia5200 import Lpneumonia5200
from federated.federated import Federated
from model.model import Model

from model.math.plot import ModelPlot

#
# FEEL FREE TO EDIT THE CONTENT OF ALL GIVEN FILES AS YOU LIKE.
#

############# APPLE SILICON (M4) GPU OPTIMIZATION ###############
def configure_metal():
    """
    Configure TensorFlow for Metal GPU acceleration on Apple Silicon.
    Not required if running on CPU-only or other GPU backends.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       # Reduce TF log noise
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # Avoid oneDNN conflicts with Metal

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[Metal] GPU acceleration enabled: {gpus}")
    else:
        print("[Metal] No GPU found — running on CPU only")

    # M4 base: 4 performance cores. Adjust for M4 Pro (8) / M4 Max (10).
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

configure_metal()
###############################################################

############# REPRODUCIBILITY, deterministic behavior #############
def set_seeds(SEED):
    """
    Set seeds for deterministic behavior across random number generators.
    Ensures reproducible results across runs and different hardware.
    """
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

def set_global_determinism(SEED):
    set_seeds(SEED=SEED)

SEED = 42   # Change seed as you like.
set_global_determinism(SEED=SEED)
###############################################################

class ModelSimulation():
    """
    Task 1: Single Model Training (Baseline).
    
    Trains a CNN model on merged datasets without federated learning.
    Used to verify the model and dataset pipeline work correctly before 
    moving to federated learning (Task 2).
    
    This is a baseline test, not part of the main assignment deliverables.
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
    Task 2: Federated Learning Phase 1 (Pre-training).
    
    Implements the core federated learning simulation described in section 2.2.2 and 3.2:
    - Initialize global and local models
    - Pre-train on IN-DISTRIBUTION data only (OOD disabled)
    - Perform Federated Averaging aggregation across rounds
    - Save trained models for use in Phase 2 (Task 3-4)
    
    Configuration below can be modified to change simulation parameters
    (rounds, clients, participants, datasets, etc.)
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
        loss = 'categorical_crossentropy'
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
    """
    Task 3: Federated Learning Phase 2 with OOD Detection.
    
    Extends FederatedSimulation with Hyperdimensional Computing (HDFF) for OOD detection.
    - Loads pre-trained models from Phase 1
    - Runs additional rounds with OOD detection ENABLED
    - Introduces one malicious local model with poisoned/OOD data
    - Demonstrates that OOD detection can filter out malicious updates
    - Related to section 3.3 (Task 3) with preliminary single OOD client scenario
    
    Only relevant configs are overridden; others inherited from FederatedSimulation.
    """
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

# ============================================================
# TASK 4: Experimentation
# ============================================================
# Task 4 (Section 3.4): Runs multiple OOD scenarios to evaluate robustness
# Scenarios include: baseline, full poisoning, partial poisoning, novel dataset OOD
#
# QUICK_TEST enables fast testing with fewer rounds; set False for full experiments
# 
# Experiment structure:
# 1. Pre-training: All clients train on clean ID data for PRETRAIN_ROUNDS
# 2. Each experiment: Load pre-trained models and run additional rounds
#    with OOD detection on and potentially malicious clients introduced

# Set QUICK_TEST = True for fast validation runs, False for full experiments
QUICK_TEST = False

PRETRAIN_PATH = "./.env/.saved/task4_pretrain/" if not QUICK_TEST else "./.env/.saved/task4_quicktest/"
PRETRAIN_ROUNDS = 35 if not QUICK_TEST else 3
EXP_EXTRA_ROUNDS = 5 if not QUICK_TEST else 2

class Task4Base:
    """
    Base class for Task 4 experiments.
    
    Provides common setup and run() method that:
    - Creates dataset and federated environment
    - Runs FL simulation
    - Generates confusion matrix and results
    - Saves plots for analysis
    
    Subclasses (Task4PreTrain, Task4Exp1-5) override configurations
    and _make_dataset() to implement specific scenarios.
    """
    experiment_name = ""
    federated_config = None
    ood_config = None

    model_config = ConfigModel(
        debug=True, epochs=1, activation='relu',
        activation_out='softmax', optimizer='adam',
        loss='categorical_crossentropy'
    )
    dataset_config = ConfigDataset(
        debug=False, batch_size=64, image_size=256,
        input_shape=(256, 256, 1), split=0.25, number_of_classes=2
    )
    plot_config = ConfigPlot(plot=True, path='./.env/plot', img_per_class=10)

    def _make_dataset(self):
        raise NotImplementedError

    def run(self):
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix as compute_cm
        import seaborn as sns

        print(f"\n{'='*60}")
        print(f"  Running: {self.experiment_name}")
        print(f"{'='*60}")

        m = Model(
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        dataset = self._make_dataset()
        federated = Federated(
            dataset=dataset, model=m,
            federated_config=self.federated_config,
            ood_config=self.ood_config,
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        federated.experiment_name = self.experiment_name
        federated.run()

        # Generate and save confusion matrix for global model
        host_id = int(self.federated_config.host_id)
        global_model = federated.models[host_id]
        _, _, test_data = federated.client_data[host_id]
        save_dir = os.path.join("./.env/plot", self.experiment_name)
        os.makedirs(save_dir, exist_ok=True)

        y_pred = global_model.model.predict(test_data)
        y_pred_classes = np.argmax(y_pred, axis=1)
        true_classes = test_data.classes
        cm = compute_cm(true_classes, y_pred_classes)

        # Append confusion matrix to results.txt
        class_names = list(test_data.class_indices.keys())
        results_path = os.path.join(save_dir, "results.txt")
        with open(results_path, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write("Confusion Matrix\n")
            f.write(f"{'='*60}\n\n")

            # Header row
            col_width = max(len(c) for c in class_names) + 4
            f.write(f"{'Predicted →':<{col_width}}")
            for name in class_names:
                f.write(f"{name:<{col_width}}")
            f.write("\n")
            f.write(f"{'Actual ↓':<{col_width}}")
            f.write(f"{'-'*col_width * len(class_names)}\n")

            for i, row_name in enumerate(class_names):
                f.write(f"{row_name:<{col_width}}")
                for j in range(len(class_names)):
                    f.write(f"{cm[i][j]:<{col_width}}")
                f.write("\n")

            # Per-class accuracy
            f.write(f"\nPer-class accuracy:\n")
            for i, name in enumerate(class_names):
                total = cm[i].sum()
                correct = cm[i][i]
                pct = (correct / total * 100) if total > 0 else 0
                f.write(f"  {name}: {correct}/{total} ({pct:.1f}%)\n")

            overall = np.trace(cm) / cm.sum() * 100
            f.write(f"\n  Overall: {np.trace(cm)}/{cm.sum()} ({overall:.1f}%)\n")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                    xticklabels=test_data.class_indices.keys(),
                    yticklabels=test_data.class_indices.keys())
        plt.xlabel('Predicted label')
        plt.ylabel('Actual label')
        plt.title(f'{self.experiment_name} - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close('all')

        print(f"  Plots saved to {save_dir}/")
        return federated


class Task4PreTrain(Task4Base):
    """Pre-train Phase 1 models for 35 rounds on ID data only. OOD disabled."""
    experiment_name = "task4_pretrain"
    federated_config = ConfigFederated(
        debug=True, save=True, load=False, load_round=0, load_reg=True,
        path=PRETRAIN_PATH,
        rounds=PRETRAIN_ROUNDS, clients=5, participants=4, host_id=0,
        client_to_dataset=[[0, 1, 2, 3], [0], [1], [2], [3]],
        ood_round=PRETRAIN_ROUNDS + 1, delete_on_load=False,
    )
    ood_config = ConfigOod(
        debug=True, hdc_debug=False, enabled=False,
        hyper_size=int(1e4), id_client=[1, 2, 3, 4], ood_client=[],
        ood_protection=False, ood_protection_thres=0.5,
    )

    def _make_dataset(self):
        return Dataset([
            (Btumor4600().ID, Btumor4600(), []),
            (Btumor3000().ID, Btumor3000(), []),
            (Balzheimer5100().ID, Balzheimer5100(), []),
            (Lpneumonia5200().ID, Lpneumonia5200(), []),
        ], dataset_config=self.dataset_config, plot_config=self.plot_config)


class Task4Exp1(Task4Base):
    """Exp 1: 1 OOD local model (complete poisoned), OOD detection DISABLED.
    Expected: global accuracy deteriorates."""
    experiment_name = "task4_exp1_ood_disabled"
    federated_config = ConfigFederated(
        debug=True, save=False, load=True, load_round=PRETRAIN_ROUNDS,
        load_reg=True, path=PRETRAIN_PATH,
        rounds=PRETRAIN_ROUNDS + EXP_EXTRA_ROUNDS, clients=2, participants=1, host_id=0,
        client_to_dataset=[[0, 1, 2, 3], [4]],
        ood_round=PRETRAIN_ROUNDS + EXP_EXTRA_ROUNDS + 1, delete_on_load=False,
    )
    ood_config = ConfigOod(
        debug=True, hdc_debug=False, enabled=False,
        hyper_size=int(1e4), id_client=[], ood_client=[1],
        ood_protection=False, ood_protection_thres=0.5,
    )

    def _make_dataset(self):
        return Dataset([
            (Btumor4600().ID, Btumor4600(), []),
            (Btumor3000().ID, Btumor3000(), []),
            (Balzheimer5100().ID, Balzheimer5100(), []),
            (Lpneumonia5200().ID, Lpneumonia5200(), []),
            (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), []),  # 4: full poisoned
        ], dataset_config=self.dataset_config, plot_config=self.plot_config)


class Task4Exp2(Task4Base):
    """Exp 2: 1 OOD local model (complete poisoned), OOD detection ENABLED.
    Expected: OOD client detected and excluded, accuracy stays stable."""
    experiment_name = "task4_exp2_ood_enabled"
    federated_config = ConfigFederated(
        debug=True, save=False, load=True, load_round=PRETRAIN_ROUNDS,
        load_reg=True, path=PRETRAIN_PATH,
        rounds=PRETRAIN_ROUNDS + EXP_EXTRA_ROUNDS, clients=2, participants=1, host_id=0,
        client_to_dataset=[[0, 1, 2, 3], [4]],
        ood_round=PRETRAIN_ROUNDS + 1, delete_on_load=False,
    )
    ood_config = ConfigOod(
        debug=True, hdc_debug=False, enabled=True,
        hyper_size=int(1e4), id_client=[], ood_client=[1],
        ood_protection=True, ood_protection_thres=0.5,
    )

    def _make_dataset(self):
        return Dataset([
            (Btumor4600().ID, Btumor4600(), []),
            (Btumor3000().ID, Btumor3000(), []),
            (Balzheimer5100().ID, Balzheimer5100(), []),
            (Lpneumonia5200().ID, Lpneumonia5200(), []),
            (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), []),  # 4: full poisoned
        ], dataset_config=self.dataset_config, plot_config=self.plot_config)


class Task4Exp3(Task4Base):
    """Exp 3: 4 ID locals + 1 new OOD local (complete poisoned), OOD ENABLED.
    Expected: OOD client 5 detected and excluded."""
    experiment_name = "task4_exp3_new_ood_poisoned"
    federated_config = ConfigFederated(
        debug=True, save=False, load=True, load_round=PRETRAIN_ROUNDS,
        load_reg=True, path=PRETRAIN_PATH,
        rounds=PRETRAIN_ROUNDS + EXP_EXTRA_ROUNDS, clients=6, participants=5, host_id=0,
        client_to_dataset=[[0, 1, 2, 3], [0], [1], [2], [3], [4]],
        ood_round=PRETRAIN_ROUNDS + 1, delete_on_load=False,
    )
    ood_config = ConfigOod(
        debug=True, hdc_debug=False, enabled=True,
        hyper_size=int(1e4), id_client=[1, 2, 3, 4], ood_client=[5],
        ood_protection=True, ood_protection_thres=0.5,
    )

    def _make_dataset(self):
        return Dataset([
            (Btumor4600().ID, Btumor4600(), []),
            (Btumor3000().ID, Btumor3000(), []),
            (Balzheimer5100().ID, Balzheimer5100(), []),
            (Lpneumonia5200().ID, Lpneumonia5200(), []),
            (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), []),  # 4: full poisoned
        ], dataset_config=self.dataset_config, plot_config=self.plot_config)


class Task4Exp4(Task4Base):
    """Exp 4: 4 ID locals + 1 new OOD local (half poisoned + half ID), OOD ENABLED.
    Expected: partially poisoned client still detected."""
    experiment_name = "task4_exp4_half_poisoned"
    federated_config = ConfigFederated(
        debug=True, save=False, load=True, load_round=PRETRAIN_ROUNDS,
        load_reg=True, path=PRETRAIN_PATH,
        rounds=PRETRAIN_ROUNDS + EXP_EXTRA_ROUNDS, clients=6, participants=5, host_id=0,
        client_to_dataset=[[0, 1, 2, 3], [0], [1], [2], [3], [4, 5]],
        ood_round=PRETRAIN_ROUNDS + 1, delete_on_load=False,
    )
    ood_config = ConfigOod(
        debug=True, hdc_debug=False, enabled=True,
        hyper_size=int(1e4), id_client=[1, 2, 3, 4], ood_client=[5],
        ood_protection=True, ood_protection_thres=0.5,
    )

    def _make_dataset(self):
        return Dataset([
            (Btumor4600().ID, Btumor4600(), []),
            (Btumor3000().ID, Btumor3000(), []),
            (Balzheimer5100().ID, Balzheimer5100(), []),
            (Lpneumonia5200().ID, Lpneumonia5200(), []),
            (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), []),          # 4: full poisoned
            (Lpneumonia5200().ID, Lpneumonia5200(), [[0, 500], [2600, 3100]]),     # 5: ~1000 ID samples
        ], dataset_config=self.dataset_config, plot_config=self.plot_config)


class Task4Exp5(Task4Base):
    """Exp 5: 4 ID locals + 1 new OOD local (Animal Faces dataset), OOD ENABLED.
    Expected: completely different dataset client detected as OOD."""
    experiment_name = "task4_exp5_animal_faces"
    federated_config = ConfigFederated(
        debug=True, save=False, load=True, load_round=PRETRAIN_ROUNDS,
        load_reg=True, path=PRETRAIN_PATH,
        rounds=PRETRAIN_ROUNDS + EXP_EXTRA_ROUNDS, clients=6, participants=5, host_id=0,
        client_to_dataset=[[0, 1, 2, 3], [0], [1], [2], [3], [4]],
        ood_round=PRETRAIN_ROUNDS + 1, delete_on_load=False,
    )
    ood_config = ConfigOod(
        debug=True, hdc_debug=False, enabled=True,
        hyper_size=int(1e4), id_client=[1, 2, 3, 4], ood_client=[5],
        ood_protection=True, ood_protection_thres=0.5,
    )

    def _make_dataset(self):
        return Dataset([
            (Btumor4600().ID, Btumor4600(), []),
            (Btumor3000().ID, Btumor3000(), []),
            (Balzheimer5100().ID, Balzheimer5100(), []),
            (Lpneumonia5200().ID, Lpneumonia5200(), []),
            (Afaces16000().ID, Afaces16000(), [[1, 700], [2501, 3200]]),  # 4: Animal faces subset
        ], dataset_config=self.dataset_config, plot_config=self.plot_config)


if __name__ == "__main__":
    TASK = 4  # Change to 1, 2, 3, or 4

    if TASK == 1:
        ModelSimulation().run()
    elif TASK == 2:
        FederatedSimulation().run()
    elif TASK == 3:
        Task3Simulation().run()
    elif TASK == 4:
        if QUICK_TEST:
            print(f"\n{'='*60}")
            print("  QUICK TEST MODE (pretrain={}, +2 rounds per exp)".format(PRETRAIN_ROUNDS))
            print(f"{'='*60}")

        experiments = [
            Task4PreTrain(),
            Task4Exp1(),
            Task4Exp2(),
            Task4Exp3(),
            Task4Exp4(),
            Task4Exp5(),
        ]
        results = {}
        for exp in experiments:
            fed = exp.run()
            results[exp.experiment_name] = fed

        # Write combined summary
        summary_path = "./.env/plot/task4_summary.txt"
        with open(summary_path, "w") as f:
            from datetime import datetime
            f.write(f"Task 4 — Experiment Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {'QUICK TEST' if QUICK_TEST else 'FULL'}\n")
            f.write(f"Pretrain rounds: {PRETRAIN_ROUNDS}, Extra rounds/exp: {EXP_EXTRA_ROUNDS}\n")
            f.write(f"{'='*70}\n\n")

            f.write(f"{'Experiment':<35}{'Final Acc':<12}{'Final Loss':<12}{'OOD':<8}\n")
            f.write(f"{'-'*70}\n")
            for name, fed in results.items():
                acc = f"{fed.global_acc_hist[-1]:.4f}" if fed.global_acc_hist else "—"
                loss = f"{fed.global_loss_hist[-1]:.4f}" if fed.global_loss_hist else "—"
                ood = "ON" if fed.ood_config.enabled else "OFF"
                f.write(f"{name:<35}{acc:<12}{loss:<12}{ood:<8}\n")

            f.write(f"\nDetailed results per experiment: ./.env/plot/<experiment>/results.txt\n")

        print(f"\n{'='*60}")
        print("  All Task 4 experiments completed!")
        print(f"  Summary:  {summary_path}")
        print("  Details:  ./.env/plot/task4_*/results.txt")
        print(f"  Plots:    ./.env/plot/task4_*/")
        print(f"{'='*60}")
