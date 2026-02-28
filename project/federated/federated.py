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
    Federated learning environment.
    Phase 1 (Task 2): initialize models + datasets, distribute weights (regression),
    train locals, aggregate (FedAvg), evaluate global, save & plot results.
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

        # Task 3: OOD tracking
        self.similarity_hist: dict[int, list[float]] = {}
        self._ood_proj = None
        self.experiment_name: str = ""

    # ------------------------------
    # Helpers (robust to Model wrapper)
    # ------------------------------
    def _keras(self, m: Model):
        """Return underlying keras model if wrapper has .model, else assume m itself behaves like keras."""
        return getattr(m, "model", m)

    def _get_weights(self, m: Model):
        km = self._keras(m)
        if hasattr(km, "get_weights"):
            return km.get_weights()
        raise AttributeError(
            "Model has no get_weights(). Expected wrapper.model.get_weights() or model.get_weights()."
        )

    def _set_weights(self, m: Model, weights):
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
    # ------------------------------
    def initialize_(self):
        """Create global + local models with unique IDs and bind datasets. Run once."""
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
        """Runs federated learning environment (Phase 1 for Task 2)."""

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
    # ------------------------------
    def train_(self, start: int):
        """
        Phase 1 training loop:
        regression -> local train -> FedAvg -> global test
        """
        host_id = int(self.federated_config.host_id)
        local_ids = [i for i in self.models.keys() if i != host_id]

        for rnd in range(1 + start, int(self.federated_config.rounds) + 1):
            part = max(int(self.federated_config.participants), 1)
            part = min(part, len(local_ids))
            selected_clients = random.sample(local_ids, part)

            # Warmup logic if OOD enabled later
            if self.ood_config.enabled and rnd < int(self.federated_config.ood_round):
                selected_clients = [cid for cid in selected_clients if cid not in self.ood_config.ood_client]
                if not selected_clients:
                    selected_clients = [random.choice([cid for cid in local_ids if cid not in self.ood_config.ood_client])]

            if self.federated_config.debug:
                print(f"\n[Round {rnd}] selected_clients={selected_clients}")

            # Task 2.2: Regression (global -> locals)
            self.global_(host_id, rnd)

            # Task 2.3: Local training
            for cid in selected_clients:
                self.local_(cid, rnd)

            # Task 3: OOD detection (filter clients before aggregation)
            if self.ood_config.enabled and rnd >= int(self.federated_config.ood_round):
                selected_clients = self.ood_detection(selected_clients, rnd)

            # Task 2.4: FedAvg aggregation (locals -> global)
            self.update_(selected_clients, rnd)

            # Global evaluation on all ID data (assigned to host)
            self.test_global_(rnd)

        return int(self.federated_config.rounds)

    def test_(self):
        return None

    # ------------------------------
    # Task 2.2: Regression (global -> locals)
    # ------------------------------
    def global_(self, id: int, rnd: int):
        """Distribute global weights to all local models."""
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
    # ------------------------------
    def local_(self, id: int, rnd: int):
        """Train local model on its assigned dataset."""
        train_data, val_data, test_data = self.client_data[id]
        self.models[id].train(train_data, val_data)

        if self.federated_config.debug:
            print(f"[Round {rnd}] Local train done: client={id}")

        return None

    # ------------------------------
    # Task 2.4: FedAvg + update global
    # ------------------------------
    def update_(self, selected_clients, rnd: int):
        """Aggregate selected local models (FedAvg) and update global model weights."""
        host_id = int(self.federated_config.host_id)
        local_weights = [self._get_weights(self.models[cid]) for cid in selected_clients]

        new_global_w = None
        if hasattr(federated_math, "fedavg"):
            try:
                new_global_w = federated_math.fedavg(local_weights)
            except Exception:
                new_global_w = None

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
    # ------------------------------
    def test_global_(self, rnd: int):
        """Evaluate global model on host's test data (should be ALL ID datasets)."""
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
    # ------------------------------
    def result(self):
        """Save plots for global accuracy/loss over rounds."""
        save_dir = os.path.join("./.env/plot", self.experiment_name) if self.experiment_name else "./.env/plot"
        os.makedirs(save_dir, exist_ok=True)

        title_prefix = f"[{self.experiment_name}] " if self.experiment_name else ""
        rounds = range(1, len(self.global_acc_hist) + 1)

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

        # Task 3: OOD similarity plot
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
    # ------------------------------
    def ood_detection(self, selected_clients, rnd: int):
        """Compare each local model to the global model using HDFF cosine similarity.
        Returns filtered list of clients that pass the OOD threshold."""
        host_id = int(self.federated_config.host_id)
        global_keras = self._keras(self.models[host_id])

        # Build HDFF signature for global model
        hdff = Hdff(self.ood_config, self.dataset_config)
        hdff.feature_extraction(global_keras)
        hdff.feature_update(global_keras)

        # Create projection matrices once, reuse across rounds
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

            # Track similarity history for plotting
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

        # Safety: don't exclude everyone
        if not filtered:
            if self.federated_config.debug:
                print(f"[OOD] Round {rnd}: All clients excluded! Keeping all to avoid empty aggregation.")
            return selected_clients

        return filtered