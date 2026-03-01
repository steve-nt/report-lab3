# Import TensorFlow for deep learning and mathematical operations
import tensorflow as tf

# Define the VSA (Vector Symbolic Architectures) class for hyperdimensional computing operations
class Vsa:
    """ VSA/HDC (Vector Symbolic Architectures / Hyperdimensional Computing) mathematics
    form the backbone of the OOD detection approach used in federated learning.
    
    References:
        - https://arxiv.org/abs/2112.05341
        - https://github.com/SamWilso/HDFF_Official
        
    However code is re-written to fit tensorflow and this project structure.
    """
    # Flag to enable debug mode for detailed output
    debug = False
    
    def __init__(self, debug) -> None:
        """
        Initialize VSA helper with debug mode flag.
        
        Parameters:
            debug (bool): Enable debug logging
        """
        # Store debug flag for controlling verbose output
        self.debug = debug
    
    def _dim_check(self, x, y):
        """
        Ensure both input tensors have at least 2 dimensions (batch, features).
        Adds batch dimension if needed.
        
        Parameters:
            x (tf.Tensor): First tensor
            y (tf.Tensor): Second tensor
            
        Returns:
            tuple: (x, y) with guaranteed shape (batch, features)
        """
        # If x is 1D, add batch dimension to make it 2D
        if len(tf.shape(x)) < 2:
            x = tf.expand_dims(x, axis=0)
        # If y is 1D, add batch dimension to make it 2D
        if len(tf.shape(y)) < 2:
            y = tf.expand_dims(y, axis=0)
        # Return both tensors with proper dimensionality
        return x, y

    def bundle(self, x, y) -> tf.Tensor:
        """
        Perform bundling operation (superposition) in hyperdimensional space.
        Bundling combines two hypervectors by simple addition.
        
        Parameters:
            x (tf.Tensor): First hypervector
            y (tf.Tensor): Second hypervector
            
        Returns:
            tf.Tensor: Bundled result (x + y) representing combined information
        """
        # Ensure both tensors have compatible dimensions
        x, y = self._dim_check(x, y)
        # Perform bundling via element-wise addition
        return x + y
	
    def bulk_bundle(self, x) -> tf.Tensor:
        """
        Perform bundling on multiple vectors by summing across axis 0 (batch).
        
        Parameters:
            x (tf.Tensor): Stack of vectors with shape (n, features)
            
        Returns:
            tf.Tensor: Sum of all vectors across batch dimension
        """
        # Sum all vectors in the batch dimension to create a single bundled vector
        return tf.reduce_sum(x, axis=0)
    
    def bind(self, x, y) -> tf.Tensor:
        """
        Perform binding operation (element-wise multiplication) in hyperdimensional space.
        Binding is used for semantic association between hypervectors.
        
        Parameters:
            x (tf.Tensor): First hypervector
            y (tf.Tensor): Second hypervector
            
        Returns:
            tf.Tensor: Bound result (x * y) representing associated information
        """
        # Ensure both tensors have compatible dimensions
        x, y = self._dim_check(x, y)
        # Perform binding via element-wise multiplication
        return x * y

    def norm(self, tensor):
        """
        Normalize tensor using L2 normalization (unit norm).
        This ensures vectors have consistent magnitude for fair similarity comparisons.
        
        Parameters:
            tensor (tf.Tensor): Input tensor to normalize
            
        Returns:
            tf.Tensor: Normalized tensor with L2 norm = 1
        """
        # Apply L2 normalization along the feature axis (axis 1) to normalize magnitude
        return tf.nn.l2_normalize(tensor, axis=1)  

    def similarity(self, x, y):
        """
        Compute cosine similarity between two hypervectors.
        Similarity ranges from -1 to 1 (or 0 to 1 for normalized vectors).
        High similarity indicates similar models/features.
        
        Parameters:
            x (tf.Tensor): First hypervector(s)
            y (tf.Tensor): Second hypervector(s)
            
        Returns:
            tf.Tensor: Cosine similarity matrix of shape (n_samples_x, n_samples_y)
        """
        # Both should be (n_samples, hyper_dim) after dimension check
        # Ensure both tensors have compatible batch dimensions
        x, y = self._dim_check(x, y)
        # Normalize both vectors to unit norm
        x, y = self.norm(x), self.norm(y)
        # Compute cosine similarity via matrix multiplication: x · y^T
        return tf.linalg.matmul(x, y, transpose_b=True)
    
    def euclidean_distance(self, x, y):
        """
        Compute pairwise Euclidean distance between two sets of vectors.
        Distance metric useful for clustering and nearest-neighbor analysis.

        Args:
            x (tf.Tensor): Tensor of shape (n_samples, n_features).
            y (tf.Tensor): Tensor of shape (m_samples, n_features).

        Returns:
            tf.Tensor: Pairwise Euclidean distances of shape (n_samples, m_samples).
        """
        # Ensure both tensors have compatible batch dimensions
        x, y = self._dim_check(x, y)
        # Compute squared differences between all pairs of samples
        squared_diff = tf.expand_dims(x, axis=1) - tf.expand_dims(y, axis=0)
        # Sum squared differences along feature axis to get squared distances
        squared_distances = tf.reduce_sum(tf.square(squared_diff), axis=-1)
        # Take square root to get actual Euclidean distances
        distances = tf.sqrt(squared_distances)

        # Return pairwise distance matrix
        return distances

# Main block for testing VSA operations
if __name__ == "__main__": 
    # Create VSA instance with debug enabled for testing
    hdff = Vsa(debug=True)
