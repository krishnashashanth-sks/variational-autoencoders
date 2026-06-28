from model import VQVAE
from torch.utils.data import DataLoader
from dataset import train_dataset,val_dataset
import torch
import torch.optim as optim
from train import train
from utils import denormalize_img
from inference import infer_vqvae
import matplotlib.pyplot as plt

input_channels = 3
num_hiddens = 128
num_downsamples = 2 # This means two stride-2 convolutions initially
num_embeddings = 512
embedding_dim = num_hiddens # Using num_hiddens from previous Encoder test
commitment_cost = 0.25 # A common value

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

vqvae_model = VQVAE(
    input_channels=input_channels,
    num_hiddens=num_hiddens,
    num_downsamples=num_downsamples,
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    commitment_cost=commitment_cost
).to(device)

train_batch_size = 64
val_batch_size = 32
num_workers_value = 2 # Adjust based on your system (e.g., number of CPU cores)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers_value)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers_value)


learning_rate = 1e-3
optimizer = optim.Adam(vqvae_model.parameters(), lr=learning_rate)
print(f"Optimizer: {optimizer} with learning rate {learning_rate}")

num_epochs = 5 # Set a reasonable number of epochs for demonstration

train(num_epochs,vqvae_model,device,optimizer,train_dataloader,val_dataloader)

dummy_inference_image = torch.randn(1, 3, 64, 64)

# Perform inference
reconstructed_img, perp, enc_indices = infer_vqvae(vqvae_model, dummy_inference_image, device)


if reconstructed_img.shape[0] > 0:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(denormalize_img(dummy_inference_image))
    axes[0].set_title('Original Dummy Image')
    axes[0].axis('off')

    axes[1].imshow(denormalize_img(reconstructed_img))
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    plt.show()
else:
    print("No image to display.")

print("--- VQ-VAE inference demonstration complete ---")