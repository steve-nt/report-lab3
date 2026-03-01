import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from functools import partial
from tqdm import tqdm

from functools import partial

from ood.VSA import Vsa
from config import ConfigDataset, ConfigOod

# Implementation Checklist for Lab 3 Task 3 - OOD Detection:
# - [ ] Dummy input shape matches (batch_size, *input_shape)
# - [ ] self.features length equals number of model layers
# - [ ] Projection matrices match each layer's channel count
# - [ ] All projected features have shape (batch, hyper_size)
# - [ ] Bundled vector has shape (batch, hyper_size)
# - [ ] Similarity returns a scalar or vector consistently

class Hdff():
    """
    Hyperdimensional Feature Fusion (HDFF) for Out-of-Distribution Detection.
    
    Lab 3 Task 3: Implements OOD detection using principles of Hyperdimensional Computing (HDC).
    
    HDFF builds a compact, high-dimensional signature of a neural network by:
    1. Extracting outputs from intermediate layers (feature_extraction, feature_update)
    2. Projecting each layer's output into a shared hypervector space (projection_matrices)
    3. Bundling (superposing) these projected vectors (feature_bundle)
    4. Comparing signatures via cosine similarity (similarity)
    
    This allows comparison of two models (global vs. local) without sharing raw data or weights.
    Implements section 3.3 (Creating Projection Matrices, Feature Vectors, Projection/Bundling/Cosine Similarity)
    and Figure 12-14 from assignment.
    
    Theory references:
    - Hyperdimensional computing uses very high dimensional vectors (e.g., 1e4)
    - Bundling (addition/superposition) combines information from multiple layers
    - Cosine similarity measures how similar two models' feature signatures are
    - Low similarity indicates OOD (out-of-distribution / malicious) updates
    - Paper: https://arxiv.org/abs/2112.05341
    """
    
    def __init__(self, ood_config : ConfigOod, dataset_config : ConfigDataset):
        """
        Initialize HDFF for a specific model.
        
        Creates storage for:
        - Projection matrices (one per layer) that map layer outputs to hypervector space
        - Feature tensors (intermediate layer outputs)
        - VSA helper for bundling and similarity operations
        - Dummy input for feature extraction without real data
        
        Args:
            ood_config (ConfigOod): Hyperdimensional configuration (hyper_size, debug flags)
            dataset_config (ConfigDataset): Dataset configuration (batch_size, input_shape)
        """
        self.vsa = Vsa(debug=ood_config.hdc_debug)
        self.dataset_config = dataset_config
        self.ood_config = ood_config
         
        self.proj = []
        self.features = [0]
        self.results = []
        
        input_shape = (self.dataset_config.batch_size,) + self.dataset_config.input_shape
        self.dummy_input = tf.ones(input_shape)
         
    def feature_update(self, model : tf.keras.models.Sequential):
        """
        Update feature vectors with outputs from all model layers using dummy input.
        
        Section 3.3.2 (Creating Feature Vectors):
        Runs the model forward pass on a dummy input and collects outputs from each layer.
        This step actually computes the tensors needed for projection and bundling.
        
        Implementation:
        - Iterates through model layers sequentially
        - For each layer matching self.layers, stores its output
        - Creates activations without needing real training data
        
        Args:
            model (tf.keras.models.Sequential): Model for extracting output feature vectors.
        """
        x = self.dummy_input
        results = []
        for layer in model.layers:
            x = layer(x, training=False)
            if layer in self.layers:
                results.append(x)

        for i, r in enumerate(results):
            self.features[i] = r

        if self.ood_config.hdc_debug:
            print(f"[HDFF] feature_update: extracted {len(self.features)} tensors")
            for i, f in enumerate(self.features):
                print(f"  [{i}] shape={f.shape}")

            
    def feature_extraction(self, model : tf.keras.models.Sequential):
        """
        Identify and prepare storage for layer outputs (preparation, no data flow).
        
        Section 3.3.2 (Creating Feature Vectors) - preparation step:
        Inspects model structure to determine how many layers we'll extract.
        Allocates self.features as a list of correct length.
        
        This is separate from feature_update because we need to know structure
        before running data through the model.
        
        Args:
            model (tf.keras.models.Sequential): Model to analyze for layer structure.
        """
        self.layers = model.layers[1:]  # Skip InputLayer
        self.features = [None] * len(self.layers)

        if self.ood_config.hdc_debug:
            print(f"[HDFF] feature_extraction: {len(self.layers)} layers identified")
            for i, layer in enumerate(self.layers):
                print(f"  [{i}] {layer.name} -> {layer.output_shape}")
        
    def feature_bundle(self, debug : bool):
        """
        Project features and bundle them into a single hypervector signature.
        
        Section 3.3.3 (Projection, Bundling & Cosine Similarity) - projection+bundling step:
        This is the core fusion step that creates a model signature.
        
        Process:
        1. For each layer's feature:
           - Reduce spatial dimensions (e.g., 4D conv output -> 2D via global avg pool)
           - Project into hypervector space using the layer's projection matrix
        2. Bundle all projected vectors using VSA addition (superposition)
        3. Average over batch to get single representative vector
        
        Result: A (1, hyper_size) vector that represents the entire model.
        
        Args:
            debug (bool): Print debug information during bundling.
            
        Returns:
            tensor: Bundled hypervector of shape (1, hyper_size).
        """
        bundle = None
        for i, (feature, proj) in enumerate(zip(self.features, self.proj)):
            # Reduce spatial dims for conv layers (4D -> 2D)
            if len(feature.shape) == 4:
                feature = tf.reduce_mean(feature, axis=[1, 2])  # global avg pool -> (batch, channels)

            projected = tf.matmul(tf.cast(feature, tf.float32), tf.cast(proj, tf.float32))  # (batch, hyper_size)

            if bundle is None:
                bundle = projected
            else:
                bundle = self.vsa.bundle(bundle, projected)

            if debug:
                print(f"  [HDFF] layer {i}: feature {feature.shape} -> projected {projected.shape}")

        # Average over batch to get a single representative vector
        bundle = tf.reduce_mean(bundle, axis=0, keepdims=True)  # (1, hyper_size)

        if debug:
            print(f"[HDFF] bundle shape: {bundle.shape}")

        return bundle
        
    def projection_matrices(self):
        """
        Create projection matrices for each layer (section 3.3.1).
        
        Each layer has different output dimensionality. To combine them in hypervector space,
        we project each into the same high-dimensional space (hyper_size).
        
        Process:
        1. For each layer feature, determine its channel dimension
        2. Create random projection matrix of shape (channels, hyper_size)
        3. Normalize columns for numerical stability
        4. Store for later reuse
        
        These matrices are fixed (non-trainable) for consistent comparisons.
        Related to Figure 12 from assignment.
        """
        self.proj = []
        for i, feature in enumerate(self.features):
            channels = int(feature.shape[-1])
            # Random normal projection, normalized columns for stability
            proj_matrix = tf.random.normal(shape=(channels, self.ood_config.hyper_size))
            proj_matrix = tf.nn.l2_normalize(proj_matrix, axis=0)
            self.proj.append(proj_matrix)

        if self.ood_config.hdc_debug:
            print(f"[HDFF] projection_matrices: created {len(self.proj)} matrices")
            for i, p in enumerate(self.proj):
                print(f"  [{i}] shape={p.shape}")

    def similarity(self, bundle1, bundle2):
        """
        Compute cosine similarity between two bundled hypervectors.
        
        Section 3.3.3 (Cosine Similarity) - final comparison step:
        Measures how similar two models' signatures are.
        High similarity = models likely both ID (in-distribution)
        Low similarity = one model likely OOD (out-of-distribution / malicious)
        
        Uses normalization to focus on direction rather than magnitude.
        Related to Figure 11 and Figure 14 from assignment.
        
        Args:
            bundle1 (tensor): Feature bundle from model 1 (e.g., global model)
            bundle2 (tensor): Feature bundle from model 2 (e.g., local model)
            
        Returns:
            tensor: Scalar cosine similarity value between 0 and 1.
        """
        sim = self.vsa.similarity(bundle1, bundle2)
        return tf.reduce_max(sim)

    def set_projection_matrices(self, proj):
        """
        Set projection matrices to externally provided ones.
        
        Used in federated setting where all clients use the same projection basis
        to compare against the global model. Ensures consistent hypervector space.
        
        Args:
            proj (list): List of projection matrices to use.
        """
        self.proj = proj

    def set_dummy_input(self, dummy_input):
        """
        Set custom dummy input tensor for feature extraction.
        
        Allows changing batch size or input shape if needed.
        
        Args:
            dummy_input (tensor): Dummy input of shape (batch_size, *input_shape)
        """
        self.dummy_input = dummy_input