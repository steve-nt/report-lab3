import tensorflow as tf
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import os

from config import ConfigDataset, ConfigFederated, ConfigOod, ConfigPlot
from dataset.dataset import Dataset
from federated.math import federated_math

from ood.hdff import Hdff
from model.model import Model


class Federated:
    """
    Federated Learning Environment for Lab 3.
    
    Implements the core FL simulation with two phases:
    
    Phase 1 (Task 2): Pre-training
    - Initialize global and local models with unique IDs
    - Distribute global weights to locals (Regression, section 2.2.2)
    - Train local models on assigned data (Local training, section 2.2.3)
    - Aggregate via Federated Averaging (section 3.2.4)
    - Evaluate global model on all ID data (section 2.2.4)
    - Save trained models for Phase 2
    
    Phase 2 (Task 3-4): OOD Detection + Training
    - Load pre-trained models
    - Run additional training rounds with OOD detection enabled
    - Before aggregation, compare each local model to global using HDFF (section 3.3)
    - Exclude clients with similarity below threshold (OOD clients)
    - Only benign (ID) clients contribute to global model updates
    
    Related to assignment section 2.2 (Design & Requirements).
    """

    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        federated_config: ConfigFederated,
        ood_config: ConfigOod,
        dataset_config: ConfigDataset,
        plot_config: ConfigPlot,
    ) -> None:
        self.dataset = dataset
        self.init_model = model  # template model (architecture/config)
        self.federated_config = federated_config
        self.ood_config = ood_config
        self.dataset_config = dataset_config
        self.plot_config = plot_config

        # Will be created in Task 2.1 (initialize)
        self.models: dict[int, Model] = {}
        self.client_data: dict[int, tuple] = {}  # id -> (train_data, val_data, test_data)

        # Track global performance (Task 2.5)
        self.global_loss_hist: list[float] = []
        self.global_acc_hist: list[float] = []

        # Task 3: OOD tracking (stores cosine similarity per client per round)
        self.similarity_hist: dict[int, list[float]] = {}
        self._ood_proj = None
        self.experiment_name: str = ""

    # ------------------------------
    # Helpers (robust to Model wrapper)
    # Helper methods to interact with Model objects and extract/set weights.
    # These abstract away whether we're using a Model wrapper or raw Keras model.
    # ------------------------------
    def _keras(self, m: Model):
        """Return underlying keras model if wrapper has .model, else assume m itself behaves like keras."""
        return getattr(m, "model", m)

    def _get_weights(self, m: Model):
        """Extract weights from a Model wrapper. Used in Task 2.4 (FedAvg aggregation)."""
        km = self._keras(m)
        if hasattr(km, "get_weights"):
            return km.get_weights()
        raise AttributeError(
            "Model has no get_weights(). Expected wrapper.model.get_weights() or model.get_weights()."
        )

    def _set_weights(self, m: Model, weights):
        """Set weights to a Model wrapper. Used in Task 2.2 (Regression) and Task 2.4 (Aggregation)."""
        km = self._keras(m)
        if hasattr(km, "set_weights"):
            km.set_weights(weights)
            return
        raise AttributeError(
            "Model has no set_weights(). Expected wrapper.model.set_weights() or model.set_weights()."
        )

    def _clone_model(self) -> Model:
        """Create a completely separate instance of the template model."""
        return copy.deepcopy(self.init_model)

    def _get_data_for_client(self, client_id: int):
        """
        Use federated_config.client_to_dataset mapping.
        Expects Dataset.get(indices) to return (train, val, test) merged across those indices.
        """
        indices = self.federated_config.client_to_dataset[client_id]
        return self.dataset.get(indices)

    # ------------------------------
    # Task 2.1: Initialize
    # Section 3.2.1 from assignment: assign unique IDs, bind datasets, create model copies
    # ------------------------------
    def initialize_(self):
        """
        Create global + local models with unique IDs and bind datasets. Run once.
        
        Task 2.1 of assignment: Initialize phase
        - Assigns each model a unique client ID (host_id for global, 1..N for locals)
        - Uses client_to_dataset mapping to bind each client to their data subsets
        - Creates deep copies of the template model for independence
        """
        self.models.clear()
        self.client_data.clear()
        self.global_loss_hist.clear()
        self.global_acc_hist.clear()
        self.similarity_hist.clear()
        self._ood_proj = None

        num_clients = int(self.federated_config.clients)
        host_id = int(self.federated_config.host_id)

        # Create model instances (global + locals)
        for cid in range(num_clients):
            self.models[cid] = self._clone_model()

        # Bind datasets to each client
        for cid in range(num_clients):
            self.client_data[cid] = self._get_data_for_client(cid)

        if self.federated_config.debug:
            print("\n[Federated] Initialized models + datasets")
            print(
                f"  host_id={host_id}, clients={num_clients}, participants/round={self.federated_config.participants}"
            )
            for cid in range(num_clients):
                ds_idx = self.federated_config.client_to_dataset[cid]
                role = "GLOBAL" if cid == host_id else "LOCAL"
                print(f"  - {role} model id={cid} -> dataset_indices={ds_idx}")

    def run(self):
        """
        Runs federated learning environment (Phase 1 for Task 2).
        
        Main entry point. Orchestrates the complete FL simulation:
        1. Initialize models and datasets (Task 2.1)
        2. Load pre-trained models if enabled (Task 2 extension)
        3. Run training loop with regression -> local train -> aggregation -> test
        4. Save final models (section 2.2.2 requirement)
        5. Generate plots and results (Task 2.5)
        """

        # 1) Init (create models + bind datasets)
        self.initialize_()

        # 2) LOAD (load global + locals) - MUST happen before training
        if self.federated_config.load:
            loaded_any = False
            for cid in self.models:
                load_path = f"{self.federated_config.path}/model{cid}_round{self.federated_config.load_round}.keras"
                if os.path.exists(load_path):
                    self.models[cid].model = tf.keras.models.load_model(load_path)
                    print(f"[LOAD] Loaded model {cid} from {load_path}")
                    loaded_any = True
                else:
                    print(f"[LOAD] Missing file for model {cid}: {load_path}")

            # If you only saved global previously, you can keep training anyway,
            # but we warn clearly so it's obvious what's happening.
            if not loaded_any:
                print("[LOAD] No models were loaded. Training will start from scratch.")

        # 3) Choose start_round
        # If you loaded round R and want to continue, start from R (so next loop begins at R+1)
        start_round = int(self.federated_config.load_round) if self.federated_config.load else 0

        # Optional: regression right after load (config flag)
        if self.federated_config.load and self.federated_config.load_reg:
            self.global_(int(self.federated_config.host_id), start_round)

        # 4) Train or test
        if start_round < int(self.federated_config.rounds):
            end_round = self.train_(start_round)
        else:
            self.test_()
            end_round = start_round

        # 5) Summary
        if self.federated_config.debug:
            print(f"\n[Federated] Finished at round={end_round}")
            if self.global_acc_hist:
                print(
                    f"[Federated] Last global acc={self.global_acc_hist[-1]:.4f}, "
                    f"loss={self.global_loss_hist[-1]:.4f}"
                )

        # 6) SAVE (save global + locals) - file names match LOAD above
        if self.federated_config.save:
            os.makedirs(self.federated_config.path, exist_ok=True)

            for cid in self.models:
                save_path = f"{self.federated_config.path}/model{cid}_round{end_round}.keras"
                self._keras(self.models[cid]).save(save_path)

                if self.federated_config.debug:
                    print(f"[SAVE] Saved model {cid} -> {save_path}")

        # 7) PLOTS (Task 2.5)
        if getattr(self.plot_config, "plot", True):
            self.result()


        return None
    

    # ------------------------------
    # Task 2 loop
    # Main training loop for both Phase 1 (pre-training, OOD disabled) and Phase 2 (OOD enabled)
    # Follows Figure 8 & 9 from assignment section 2.2.2
    # ------------------------------
    def train_(self, start: int):
        """
        Phase 1 & Phase 2 training loop (section 2.2.2):
        Each round executes: regression -> local train -> OOD filter -> FedAvg -> global test
        
        This implements the sequential simulation described in section 3 Implementation,
        where each step is executed in order within a single round.
        """
        host_id = int(self.federated_config.host_id)
        local_ids = [i for i in self.models.keys() if i != host_id]

        for rnd in range(1 + start, int(self.federated_config.rounds) + 1):
            part = max(int(self.federated_config.participants), 1)
            part = min(part, len(local_ids))
            selected_clients = random.sample(local_ids, part)

            # Warmup logic if OOD enabled later (Phase 2)
            # Before ood_round, exclude OOD clients from training
            if self.ood_config.enabled and rnd < int(self.federated_config.ood_round):
                selected_clients = [cid for cid in selected_clients if cid not in self.ood_config.ood_client]
                if not selected_clients:
                    selected_clients = [random.choice([cid for cid in local_ids if cid not in self.ood_config.ood_client])]

            if self.federated_config.debug:
                print(f"\n[Round {rnd}] selected_clients={selected_clients}")

            # Task 2.2: Regression (global -> locals) - distribute global weights
            self.global_(host_id, rnd)

            # Task 2.3: Local training - each selected client trains on its data
            for cid in selected_clients:
                self.local_(cid, rnd)

            # Task 3: OOD detection (Phase 2 only) - filter clients before aggregation
            # Uses HDFF to detect OOD local models by comparing feature signatures
            if self.ood_config.enabled and rnd >= int(self.federated_config.ood_round):
                selected_clients = self.ood_detection(selected_clients, rnd)

            # Task 2.4: FedAvg aggregation (locals -> global) - combine trained models
            self.update_(selected_clients, rnd)

            # Global evaluation on all ID data (assigned to host)
            # Task 2.5: Test and record global model performance
            self.test_global_(rnd)

        return int(self.federated_config.rounds)

    def test_(self):
        return None

    # ------------------------------
    # Task 2.2: Regression (global -> locals)
    # Section 3.2.2 from assignment: distribute global model weights to all local models
    # This synchronizes local models with current global state before local training
    # ------------------------------
    def global_(self, id: int, rnd: int):
        """
        Distribute global weights to all local models.
        
        Task 2.2 (Regression) from section 3.2.2:
        - Extract weights from global model
        - Set these weights on all local models
        - Used at the start of each round to synchronize client state
        """
        host_id = int(id)
        global_w = self._get_weights(self.models[host_id])

        for cid, m in self.models.items():
            if cid == host_id:
                continue
            self._set_weights(m, global_w)

        if self.federated_config.debug:
            print(f"[Round {rnd}] Regression: distributed global weights to locals")

        return None

    # ------------------------------
    # Task 2.3: Local training
    # Section 3.2.3 from assignment: train each local model on its assigned data
    # ------------------------------
    def local_(self, id: int, rnd: int):
        """
        Train local model on its assigned dataset.
        
        Task 2.3 (Train) from section 3.2.3:
        - Each local client trains on their subset of data
        - Training is done for model_config.epochs epochs per round
        - Models train independently without sharing data
        """
        train_data, val_data, test_data = self.client_data[id]
        self.models[id].train(train_data, val_data)

        if self.federated_config.debug:
            print(f"[Round {rnd}] Local train done: client={id}")

        return None

    # ------------------------------
    # Task 2.4: FedAvg + update global
    # Section 3.2.4 from assignment: Federated Averaging aggregation
    # Implements equation (2) from section 3.2.4
    # ------------------------------
    def update_(self, selected_clients, rnd: int):
        """
        Aggregate selected local models (FedAvg) and update global model weights.
        
        Task 2.4 (Aggregation) from section 3.2.4:
        Implements Federated Averaging (equation 2):
          w_{t+1} = sum_k (1/K * w_k^t)
        Where:
          - K = number of participating clients
          - w_k^t = weights of client k after local training
        
        Process:
        1. Extract weights from each participating local client
        2. Average weights layer-by-layer across all clients
        3. Update global model with averaged weights
        """
        host_id = int(self.federated_config.host_id)
        local_weights = [self._get_weights(self.models[cid]) for cid in selected_clients]

        new_global_w = None
        if hasattr(federated_math, "fedavg"):
            try:
                new_global_w = federated_math.fedavg(local_weights)
            except Exception:
                new_global_w = None

        # If fedavg not implemented, use fallback averaging
        if new_global_w is None:
            new_global_w = []
            for layer_idx in range(len(local_weights[0])):
                layer_stack = np.stack([lw[layer_idx] for lw in local_weights], axis=0)
                new_global_w.append(np.mean(layer_stack, axis=0))

        self._set_weights(self.models[host_id], new_global_w)

        if self.federated_config.debug:
            print(f"[Round {rnd}] Aggregation: updated global model with FedAvg over {len(selected_clients)} clients")

        return None

    # ------------------------------
    # Global evaluation each round
    # Task 2.5: Test global model and track convergence
    # Section 2.2.4 requirement: report global model accuracy and loss per round
    # ------------------------------
    def test_global_(self, rnd: int):
        """
        Evaluate global model on host's test data (should be ALL ID datasets).
        
        Task 2.5 from section 3.2.5:
        - Global model is tested on all in-distribution test data
        - Performance tracked per round to measure convergence
        - Results plotted in section 2.2.4 Figure 10
        """
        host_id = int(self.federated_config.host_id)
        _, _, test_data = self.client_data[host_id]

        self.models[host_id].test(test_data)

        # Read from model's internal tracking (model.test() stores results internally)
        acc = self.models[host_id].test_accuracy[-1] if self.models[host_id].test_accuracy else None
        loss = self.models[host_id].test_loss[-1] if self.models[host_id].test_loss else None

        if loss is not None:
            self.global_loss_hist.append(loss)
        if acc is not None:
            self.global_acc_hist.append(acc)

        if self.federated_config.debug:
            if acc is not None and loss is not None:
                print(f"[Round {rnd}] Global test: acc={acc:.4f}, loss={loss:.4f}")
            else:
                print(f"[Round {rnd}] Global test done")

    # ------------------------------
    # Task 2.5: Plot results
    # Section 2.2.4: Save plots for accuracy, loss, and OOD detection results
    # Generates Figures 10 and 11 from assignment
    # ------------------------------
    def result(self):
        """
        Save plots and text results for global accuracy/loss over rounds.
        
        Task 2.5 from section 3.2.5:
        - Generates plots of global model accuracy and loss convergence
        - For Phase 2 with OOD enabled, also plots similarity scores per client
        - Saves results to text file with per-round metrics
        - Implements section 2.2.4 (Plot, Diagrams & Producing Results)
        """
        save_dir = os.path.join("./.env/plot", self.experiment_name) if self.experiment_name else "./.env/plot"
        os.makedirs(save_dir, exist_ok=True)

        title_prefix = f"[{self.experiment_name}] " if self.experiment_name else ""
        rounds = range(1, len(self.global_acc_hist) + 1)

        # ---- Save text results ----
        results_path = os.path.join(save_dir, "results.txt")
        with open(results_path, "w") as f:
            f.write(f"{'='*60}\n")
            f.write(f"  Experiment: {self.experiment_name or 'unnamed'}\n")
            f.write(f"{'='*60}\n\n")

            # Config summary
            f.write("Configuration\n")
            f.write(f"  Rounds:            {self.federated_config.rounds}\n")
            f.write(f"  Clients:           {self.federated_config.clients}\n")
            f.write(f"  Participants/round:{self.federated_config.participants}\n")
            f.write(f"  OOD enabled:       {self.ood_config.enabled}\n")
            if self.ood_config.enabled:
                f.write(f"  OOD threshold:     {self.ood_config.ood_protection_thres}\n")
                f.write(f"  OOD protection:    {self.ood_config.ood_protection}\n")
                f.write(f"  OOD clients:       {self.ood_config.ood_client}\n")
            f.write("\n")

            # Global accuracy/loss per round
            f.write(f"{'Round':<8}{'Accuracy':<14}{'Loss':<14}\n")
            f.write(f"{'-'*36}\n")
            for rnd, acc, loss in zip(rounds, self.global_acc_hist, self.global_loss_hist):
                f.write(f"{rnd:<8}{acc:<14.4f}{loss:<14.4f}\n")

            if self.global_acc_hist:
                f.write(f"\nFinal global accuracy:  {self.global_acc_hist[-1]:.4f}\n")
                f.write(f"Final global loss:      {self.global_loss_hist[-1]:.4f}\n")

            # OOD similarity per round (Figure 11 from assignment)
            if self.similarity_hist:
                f.write(f"\n{'='*60}\n")
                f.write("OOD Cosine Similarity per Client\n")
                f.write(f"{'='*60}\n\n")

                # Header
                client_ids = sorted(self.similarity_hist.keys())
                header = f"{'Round':<8}" + "".join(f"{'Client '+str(c):<14}" for c in client_ids)
                f.write(header + "\n")
                f.write(f"{'-'*len(header)}\n")

                max_len = max(len(s) for s in self.similarity_hist.values())
                for i in range(max_len):
                    row = f"{i+1:<8}"
                    for cid in client_ids:
                        sims = self.similarity_hist[cid]
                        if i < len(sims):
                            val = sims[i]
                            marker = " *" if self.ood_config.ood_protection and val < self.ood_config.ood_protection_thres else ""
                            row += f"{val:<12.4f}{marker:>2}"
                        else:
                            row += f"{'—':<14}"
                    f.write(row + "\n")

                f.write(f"\n  * = excluded (below threshold {self.ood_config.ood_protection_thres})\n")

        plt.figure()
        plt.plot(rounds, self.global_acc_hist, marker='o')
        plt.title(f"{title_prefix}Global Accuracy over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(save_dir, "global_accuracy.png"))

        plt.figure()
        plt.plot(rounds, self.global_loss_hist, marker='o')
        plt.title(f"{title_prefix}Global Loss over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(save_dir, "global_loss.png"))

        # Task 3: OOD similarity plot (Figure 11)
        if self.similarity_hist:
            plt.figure(figsize=(10, 6))
            for cid, sims in sorted(self.similarity_hist.items()):
                label = f"Client {cid}"
                if cid in self.ood_config.ood_client:
                    label += " (OOD)"
                plt.plot(range(1, len(sims) + 1), sims, label=label, marker='o')
            plt.axhline(y=self.ood_config.ood_protection_thres, color='k', linestyle='--',
                        label=f"Threshold ({self.ood_config.ood_protection_thres})")
            plt.title(f"{title_prefix}OOD Similarity (protection={self.ood_config.ood_protection})")
            plt.xlabel("Round")
            plt.ylabel("Cosine Similarity")
            plt.legend()
            plt.savefig(os.path.join(save_dir, "ood_similarity.png"))

        plt.show()

    # ------------------------------
    # Task 3: OOD Detection
    # Section 3.3 & 3.4 from assignment: Hyperdimensional Feature Fusion (HDFF)
    # Implements Phase 2 OOD detection mechanism
    # Figure 9 from assignment shows feature extraction step
    # Figure 14 shows projection, bundling, and cosine similarity
    # ------------------------------
    def ood_detection(self, selected_clients, rnd: int):
        """
        Compare each local model to the global model using HDFF cosine similarity.
        Returns filtered list of clients that pass the OOD threshold.
        
        Task 3 (OOD Detection) from section 3.3:
        
        Process:
        1. Extract layer outputs from global model using dummy input
        2. Create/reuse projection matrices to map features to hypervector space
        3. Bundle (superpose) projected features into a single global signature
        4. For each local client:
           a. Extract its layer features
           b. Project and bundle using same projection matrices
           c. Compute cosine similarity between global and local bundles
           d. Track similarity per client per round
        5. Exclude clients below ood_protection_thres from aggregation
        
        This implements section 3.3 (Task 3) with HDFF from paper https://arxiv.org/abs/2112.05341
        """
        host_id = int(self.federated_config.host_id)
        global_keras = self._keras(self.models[host_id])

        # Build HDFF signature for global model (section 3.3.1-3.3.3)
        hdff = Hdff(self.ood_config, self.dataset_config)
        hdff.feature_extraction(global_keras)
        hdff.feature_update(global_keras)

        # Create projection matrices once, reuse across rounds (section 3.3.1)
        if self._ood_proj is None:
            hdff.projection_matrices()
            self._ood_proj = hdff.proj
        else:
            hdff.set_projection_matrices(self._ood_proj)

        global_bundle = hdff.feature_bundle(self.ood_config.hdc_debug)

        filtered = []
        for cid in selected_clients:
            local_keras = self._keras(self.models[cid])

            hdff_local = Hdff(self.ood_config, self.dataset_config)
            hdff_local.feature_extraction(local_keras)
            hdff_local.feature_update(local_keras)
            hdff_local.set_projection_matrices(self._ood_proj)

            local_bundle = hdff_local.feature_bundle(self.ood_config.hdc_debug)
            sim_val = float(hdff.similarity(global_bundle, local_bundle))

            # Track similarity history for plotting (Figure 11)
            if cid not in self.similarity_hist:
                self.similarity_hist[cid] = []
            self.similarity_hist[cid].append(sim_val)

            if self.ood_config.ood_protection and sim_val < self.ood_config.ood_protection_thres:
                if self.federated_config.debug:
                    print(f"[OOD] Round {rnd}: Client {cid} EXCLUDED (sim={sim_val:.4f} < {self.ood_config.ood_protection_thres})")
            else:
                filtered.append(cid)
                if self.federated_config.debug:
                    print(f"[OOD] Round {rnd}: Client {cid} OK (sim={sim_val:.4f})")

        # Safety: don't exclude everyone (avoid empty aggregation)
        if not filtered:
            if self.federated_config.debug:
                print(f"[OOD] Round {rnd}: All clients excluded! Keeping all to avoid empty aggregation.")
            return selected_clients

        return filtered