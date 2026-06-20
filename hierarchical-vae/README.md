# Hierarchical Variational Autoencoder (HVAE)

A PyTorch implementation of a Hierarchical Variational Autoencoder using TensorFlow/Keras. This model learns a hierarchical latent representation with two levels of latent variables (z1 and z2) for generating MNIST digit images.

## Overview

The Hierarchical VAE extends the standard VAE by introducing a two-level hierarchical structure in the latent space:
- **z1**: First-level latent variable (32-dimensional)
- **z2**: Second-level latent variable (16-dimensional)

This hierarchical structure allows the model to capture features at different levels of abstraction, potentially leading to better disentangled representations and improved generative performance.

## Project Structure

```
hierarchical-vae/
├── dataset.py      # Data loading and preprocessing
├── layers.py       # Custom layers and architecture components
├── model.py        # HVAE model class and training logic
├── main.py         # Training and inference pipeline
└── README.md       # This file
```

## Files Description

### `dataset.py`
Handles MNIST dataset loading and preprocessing:
- Loads MNIST training and test data
- Normalizes pixel values to [0, 1] range
- Reshapes data to match the model's expected input dimensions (28, 28, 1)

### `layers.py`
Defines the encoder and decoder architectures:
- **`Sampling` Layer**: Implements the reparameterization trick for sampling from latent distributions
- **`build_encoder()`**: Creates the encoder network that maps images to latent means, log-variances, and samples for both z1 and z2
- **`build_decoder()`**: Creates the decoder network that reconstructs images from concatenated z1 and z2 latent vectors

#### Encoder Architecture:
- Input: 28×28×1 grayscale images
- Conv2D layers (32→64→128→256 filters) with ReLU activation
- Flatten and Dense(512)
- Two separate pathways:
  - z1: Dense layers for mean and log-variance (32-dim)
  - z2: Dense(256) → Dense layers for mean and log-variance (16-dim)

#### Decoder Architecture:
- Input: Concatenated z1 (32-dim) + z2 (16-dim)
- Dense(7×7×256) with ReLU
- Reshape to (7, 7, 256)
- Conv2DTranspose layers for upsampling
- Output: 28×28×1 reconstructed image with sigmoid activation

### `model.py`
Implements the HVAE model class with custom training logic:
- **HVAE Class**: Extends `keras.Model` with custom `train_step()`
- **Loss Function**: Combines reconstruction loss (binary crossentropy) and KL divergence losses for both z1 and z2
- **Metrics Tracking**: Monitors total loss, reconstruction loss, and individual KL losses

#### Loss Computation:
```
Total Loss = Reconstruction Loss + KL Loss (z1) + KL Loss (z2)
```

### `main.py`
Training and inference pipeline:
- Initializes encoder and decoder
- Compiles HVAE model with Adam optimizer
- Generates 10 sample images from random latent vectors
- Visualizes generated images using matplotlib

## Configuration

Key hyperparameters defined in `main.py`:
- `IMAGE_SIZE`: (28, 28, 1) - Input image dimensions
- `LATENT_DIM_1`: 32 - First-level latent dimension
- `LATENT_DIM_2`: 16 - Second-level latent dimension
- `BATCH_SIZE`: 128 - Training batch size
- `EPOCHS`: 10 - Number of training epochs

## Requirements

```
tensorflow>=2.10.0
matplotlib>=3.5.0
numpy>=1.21.0
```

## Usage

### Training and Inference

```bash
python main.py
```

This will:
1. Build the encoder and decoder architectures
2. Instantiate the HVAE model
3. Generate 10 sample images from random latent vectors
4. Display the generated images

### Training on Your Data

To train the model on MNIST data:

```python
from dataset import x_train, x_test
from main import hvae

hvae.fit(x_train, epochs=10, batch_size=128)
```

### Generating Images

Generate new images from random latent vectors:

```python
import tensorflow as tf
z_sample_z1 = tf.random.normal(shape=(10, 32))
z_sample_z2 = tf.random.normal(shape=(10, 16))
generated_images = hvae.decoder.predict([z_sample_z1, z_sample_z2])
```

### Encoding Images

Encode images to latent representations:

```python
z1_mean, z1_log_var, z1, z2_mean, z2_log_var, z2 = hvae.encoder(x_test[:10])
```

## Model Architecture Diagram

```
Input Image (28×28×1)
        ↓
    Encoder
        ↓
    [Conv2D layers]
        ↓
    [Dense(512)]
        ↓
    ├─→ [z1_mean, z1_log_var] → Sampling → z1 (32-dim)
    └─→ [Dense(256)] → [z2_mean, z2_log_var] → Sampling → z2 (16-dim)
        ↓
    Concatenate [z1, z2]
        ↓
    Decoder
        ↓
    [Dense → Reshape]
        ↓
    [Conv2DTranspose layers]
        ↓
Reconstructed Image (28×28×1)
```

## Loss Functions

### Reconstruction Loss
Binary cross-entropy loss between original and reconstructed images:
```
L_recon = sum(binary_crossentropy(x, x_recon)) * image_size
```

### KL Divergence Loss
For each latent level (z1 and z2):
```
KL(z) = -0.5 * sum(1 + log_var - mean² - exp(log_var))
```

Total loss = L_recon + KL(z1) + KL(z2)

## Key Features

- **Hierarchical Latent Space**: Two-level hierarchy enables multi-scale feature learning
- **Reparameterization Trick**: Enables efficient gradient-based training
- **Custom Training Loop**: Full control over loss computation and optimization
- **MNIST Dataset**: Pre-built support for MNIST digit generation

## Potential Improvements

1. Add validation/test split evaluation
2. Implement checkpointing to save best models
3. Add more visualization utilities (t-SNE, latent space interpolation)
4. Support for other datasets
5. Implement beta-VAE for better disentanglement
6. Add conditional generation capabilities

## References

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.
- Burda, Y., Grosse, R., & Salakhutdinov, R. (2015). Importance Weighted Autoencoders. arXiv preprint arXiv:1509.00519.

## License

This project is open source and available under the MIT License.

## Author

Krishna Shashanth (@krishnashashanth-sks)
