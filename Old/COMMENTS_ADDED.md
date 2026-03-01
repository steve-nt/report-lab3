# Code Comments Added - Lab 3 Project

## Summary
This document outlines the detailed code comments that have been added to Python files in the lab 3 project according to the lab instructions (Section 3.2.2 and related implementation guidance).

## Files with Detailed Comments Added

### Core Configuration Files
1. **config.py** - Added comprehensive comments explaining:
   - All configuration classes (ConfigFederated, ConfigOod, ConfigModel, ConfigDataset, ConfigPlot)
   - Purpose of each configuration parameter
   - Constraints and validation rules

### Model Architecture & Training
2. **model/model.py** - Added line-by-line comments explaining:
   - Initialization of model with configurations
   - CNN architecture layers (InputLayer, Conv2D, MaxPooling2D, Flatten, Dense)
   - Training procedure (fit, epochs tracking)
   - Testing/evaluation process
   - Plotting functions for results visualization

### Main Simulation Entry Point
3. **main.py** - Added comments explaining:
   - Import statements and their purposes
   - Reproducibility and seed-setting functions
   - Model and Federated simulation classes
   - Configuration setup for different tasks

### Federated Learning Framework
4. **federated/federated.py** - Added detailed comments for:
   - Federated learning class structure and initialization
   - Model cloning and data binding
   - Phase 1 implementation (regression, local training, aggregation)
   - Test and evaluation procedures
   - Model saving/loading mechanisms

### Dataset Management
5. **dataset/dataset.py** - Added explanatory comments for:
   - Dataset class structure
   - Dataset merging functionality
   - Dataset subsetting by index
   - Index range validation

6. **dataset/generator.py** - Added detailed comments for:
   - Image loading and preprocessing pipeline
   - Train/validation/test data splitting
   - ImageDataGenerator configuration
   - Default image processing (grayscale conversion, contrast enhancement, sharpening)

### Out-of-Distribution Detection
7. **ood/hdff.py** - Added comprehensive comments explaining:
   - Hyperdimensional Feature Fusion (HDFF) concept and theory
   - Feature extraction from neural network layers
   - Projection matrix creation and dimensionality reduction
   - Feature bundling (superposition)
   - Similarity computation for OOD detection

8. **ood/VSA.py** - Added detailed comments for:
   - Vector Symbolic Architectures (VSA) operations
   - Bundling (superposition via addition)
   - Binding (association via multiplication)
   - L2 normalization
   - Cosine similarity calculation
   - Euclidean distance computation

## Key Concepts Explained

### Federated Learning
- Local and global models distribution
- Regression (weight distribution) process
- Local training on decentralized data
- FedAvg aggregation algorithm
- Global model evaluation

### Out-of-Distribution Detection
- Hyperdimensional computing principles
- Feature extraction from multiple layers
- Projection into high-dimensional space
- Similarity-based anomaly detection
- Threshold-based filtering for security

### Image Processing
- Grayscale conversion for consistency
- Contrast enhancement via weighted sums
- Sharpening with convolution kernels
- Pixel value normalization and clipping

## Lab Instructions Reference
The comments were added following the lab instructions which specify that:
1. Code should be documented with clear explanations of each line/section
2. Implementation should follow Phases 1 and 2 as described in section 2.2
3. Configuration should be modifiable for different simulation scenarios
4. Results should be tracked and plotted for analysis

## Files Not Modified
Some utility files were not extensively commented due to their brevity or straightforward nature:
- `dataset/download/*.py` - Dataset import wrappers
- `dataset/math/plot.py` - Plotting utilities
- `model/math/plot.py` - Model visualization
- `federated/math/plot.py` - Federated learning visualization
- `dataset/gen/dataframe.py` - DataFrame generation helper
- `ood/math/score.py` - OOD scoring utilities
