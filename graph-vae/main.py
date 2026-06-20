from layers import *
from model import GraphVAE
from tensorflow import keras
from dataset import generate_synthetic_graph
import matplotlib.pyplot as plt
import networkx as nx

NUM_NODES=20
NODE_FEATURES=16
BATCH_SIZE=1
EPOCHS=20
LATENT_DIM=128

encoder=build_encoder(NUM_NODES,NODE_FEATURES,LATENT_DIM)

decoder=build_decoder(NUM_NODES,LATENT_DIM)

# Instantiate and Compile the GraphVAE
graph_vae = GraphVAE(encoder, decoder)
graph_vae.compile(optimizer=keras.optimizers.Adam())

# Generate a single graph for demonstration
NODE_FEATURES_DIM = NODE_FEATURES
X_train, A_true_train, A_hat_train = generate_synthetic_graph(NUM_NODES, NODE_FEATURES_DIM)

dataset = tf.data.Dataset.from_tensors((X_train, A_true_train, A_hat_train)).repeat(EPOCHS)

print(f"Training GraphVAE for {EPOCHS} epochs...")
graph_vae.fit(dataset, epochs=EPOCHS, steps_per_epoch=1)

# Take a sample from the training data (we only have one graph here)
input_features = tf.expand_dims(X_train, axis=0)
input_adj_hat = tf.expand_dims(A_hat_train, axis=0)

# Encode the input to get latent variables
mu, log_var, z = graph_vae.encoder([input_features, input_adj_hat])

# Decode the latent variables to reconstruct the adjacency matrix
a_recon_inference = graph_vae.decoder(z)

# Remove the batch dimension for easier comparison
a_recon_inference = tf.squeeze(a_recon_inference, axis=0)

print("Original Adjacency Matrix (first 5x5):")
print(A_true_train[:5, :5])
print("\nReconstructed Adjacency Matrix (first 5x5, probabilities):")
print(a_recon_inference.numpy()[:5, :5])

# Convert reconstructed probabilities to binary adjacency matrix
threshold = 0.5
a_recon_binary = (a_recon_inference.numpy() > threshold).astype(int)

# Create NetworkX graphs
G_true = nx.from_numpy_array(A_true_train)
G_recon = nx.from_numpy_array(a_recon_binary)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
nx.draw_circular(G_true, with_labels=False, node_color='skyblue', node_size=200, edge_color='gray')
plt.title('Original Graph (Adjacency Matrix)')

plt.subplot(1, 2, 2)
nx.draw_circular(G_recon, with_labels=False, node_color='lightcoral', node_size=200, edge_color='gray')
plt.title('Reconstructed Graph (Adjacency Matrix)')

plt.show()