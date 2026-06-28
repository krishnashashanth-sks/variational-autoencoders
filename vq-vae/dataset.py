from torchvision import datasets, transforms

# 2. Define the image transformations
# CIFAR-10 images are 32x32. Resize to 64x64 as assumed in VQ-VAE model.
# Normalization stats for CIFAR-10 (mean and std for R, G, B channels)
# Note: If VQ-VAE expects input in [-1, 1], then normalize to that range.
# Typical ImageNet normalization values (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# For VQ-VAE, it's often more common to normalize to [-0.5, 0.5] or similar, then use Tanh in decoder.
# Let's target [-1, 1] as the prompt suggests, assuming the VQ-VAE decoder's Tanh output is in [-1, 1].

transform = transforms.Compose([
    transforms.Resize(64), # Resize images to 64x64
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor (scales to [0, 1])
    # Normalize to [-1, 1] range: (x - 0.5) / 0.5
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("Defined image transformations: Resize to 64x64, ToTensor, Normalize to [-1, 1].")

# 3. Download and load the CIFAR-10 training dataset
# Using root='./data' to store the dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
print(f"Loaded CIFAR-10 training dataset with {len(train_dataset)} samples.")

# 4. Download and load the CIFAR-10 test (validation) dataset
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
print(f"Loaded CIFAR-10 validation dataset with {len(val_dataset)} samples.")
