# Graph Variational Autoencoder (Graph-VAE)

A TensorFlow implementation of a Variational Autoencoder (VAE) designed for learning latent representations of graph structures using Graph Convolutional Networks (GCNs).

## Project Overview

This project implements a Graph-VAE architecture that combines:
- **Graph Convolutional Networks (GCN)** for graph encoding
- **Variational Autoencoder (VAE)** framework for unsupervised learning
- **Synthetic graph generation** for testing and demonstration

The model learns to reconstruct graph adjacency matrices from their node features and graph structure, enabling the generation of new graph structures through sampling from the learned latent space.

## Files Description

### `dataset.py`
Handles synthetic graph generation for training and testing.

**Key Functions:**
- `generate_synthetic_graph(num_nodes, node_features_dim, num_edges_factor=2)`
  - Generates random node features with shape `(num_nodes, node_features_dim)`
  - Creates a sparse random adjacency matrix
  - Normalizes the adjacency matrix using the GCN normalization: `A_hat = D^(-1/2) * (A + I) * D^(-1/2)`
  - Returns: normalized node features (X), original adjacency matrix (adj), and normalized adjacency matrix (A_hat)

### `layers.py`
Contains custom TensorFlow layers and encoder/decoder builders.

**Key Components:**
- **GCNLayer**: Custom Graph Convolutional Network layer
  - Applies spectral convolution: `output = A_hat * X * W`
  - Supports optional activation functions
  - Uses Glorot uniform initialization
  
- **Sampling**: Reparameterization trick layer
  - Samples from latent distribution N(μ, σ²)
  - Formula: `z = μ + σ * ε`, where ε ~ N(0, I)
  
- **build_encoder()**: Constructs the encoder model
  - Input: Node features and normalized adjacency matrix
  - Architecture: Input → GCN (64 units) → Latent μ and log_var → Sampling layer
  - Output: Mean (μ), log variance (log_var), and sampled latent vector (z)
  
- **build_decoder()**: Constructs the decoder model
  - Input: Sampled latent vector (z)
  - Reconstructs adjacency matrix: `A_recon = sigmoid(z * z^T)`
  - Output: Reconstructed adjacency matrix probabilities

### `model.py`
Defines the Graph VAE model combining encoder and decoder.

**Key Class:**
- **GraphVAE**: Main VAE model inheriting from `keras.Model`
  - **Metrics tracked**: Total loss, reconstruction loss, KL divergence loss
  - **train_step()**: Custom training loop
    - Reconstruction loss: Binary crossentropy between original and reconstructed adjacency matrices
    - KL divergence loss: `-0.5 * Σ(1 + log_var - μ² - exp(log_var))`
    - Total loss: `reconstruction_loss + KL_loss`
  - **call()**: Forward pass for inference
    - Encodes input graph → Samples latent vector → Decodes to reconstruct adjacency matrix

### `main.py`
Training script demonstrating the full pipeline.

**Pipeline:**
1. Initialize encoder and decoder
2. Compile GraphVAE model with Adam optimizer
3. Generate synthetic graph dataset
4. Train for specified epochs
5. Encode and decode a sample graph
6. Visualize original vs. reconstructed graphs using NetworkX

**Hyperparameters:**
- `NUM_NODES`: 20
- `NODE_FEATURES`: 16
- `BATCH_SIZE`: 1
- `EPOCHS`: 20
- `LATENT_DIM`: 128

## Installation

### Requirements
```bash
pip install tensorflow>=2.8.0
pip install numpy
pip install matplotlib
pip install networkx
```

### Setup
```bash
git clone <repository-url>
cd graph-vae
```

## Usage

### Basic Training
```bash
python main.py
```

This will:
1. Train the Graph-VAE for 20 epochs on a synthetic graph
2. Print original and reconstructed adjacency matrices
3. Display side-by-side visualization of original and reconstructed graphs

### Custom Configuration
Edit `main.py` to modify hyperparameters:
```python
NUM_NODES = 50          # Increase graph size
LATENT_DIM = 256        # Increase latent dimension
EPOCHS = 100            # Train longer
NODE_FEATURES = 32      # Increase node feature dimension
```

## Model Architecture

### Encoder Architecture
```
Node Features (N × F) ──┐
                        ├─→ GCN Layer 1 (64 units) ──┬─→ GCN Latent μ (N × L)
Adj Matrix Normalized   │                            │
    (N × N)      ────┐  │                            ├─→ GCN Latent log_var (N × L)
                     └──┘                            │
                                                     └─→ Sampling → z (N × L)
```

### Decoder Architecture
```
Latent Vector z (N × L) ──→ Adj Reconstruction (sigmoid(z * z^T)) ──→ A_recon (N × N)
```

## Loss Functions

### Reconstruction Loss
Binary crossentropy between original adjacency matrix and reconstructed probabilities:
```
L_recon = -Σ [A * log(A_recon) + (1-A) * log(1-A_recon)]
```

### KL Divergence Loss
Standard VAE KL divergence regularization:
```
L_KL = -0.5 * Σ (1 + log_var - μ² - exp(log_var))
```

### Total Loss
```
L_total = L_recon + L_KL
```

## Output

The script generates:
1. **Console output**: 
   - Training progress with loss metrics
   - Original and reconstructed adjacency matrices (first 5×5)

2. **Visualization**:
   - Side-by-side comparison of original and reconstructed graphs
   - Nodes colored differently for easy distinction
   - Circular layout for clear visualization

## Example Output

```
Training GraphVAE for 20 epochs...
Epoch 1/20 ... loss: 234.5 reconstruction_loss: 145.2 kl_loss: 89.3
...
Original Adjacency Matrix (first 5x5):
[[0. 1. 0. 1. 0.]
 [1. 0. 1. 0. 1.]
 [0. 1. 0. 1. 0.]
 [1. 0. 1. 0. 1.]
 [0. 1. 0. 1. 0.]]

Reconstructed Adjacency Matrix (first 5x5, probabilities):
[[0.12 0.87 0.05 0.89 0.08]
 [0.88 0.15 0.91 0.12 0.85]
 [0.06 0.92 0.14 0.88 0.09]
 [0.90 0.10 0.86 0.18 0.82]
 [0.07 0.85 0.11 0.81 0.16]]
```

## Future Enhancements

- [ ] Support for node attributes in reconstruction
- [ ] Sampling from latent space to generate new graphs
- [ ] Graph generation with specific properties
- [ ] Support for directed graphs
- [ ] Attention mechanisms in GCN layers
- [ ] Evaluation metrics (graph similarity, structural properties)

## References

- Kipf, T., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational Bayes.
- Simonovsky, M., & Komodakis, N. (2018). GraphVAE: towards generation of small graphs using variational autoencoders.

## License

This project is part of the Variational Autoencoders repository.

## Author

krishnashashanth-sks
