import torch

def infer_vqvae(model, image_tensor, device):
    """
    Performs inference with the VQ-VAE model.

    Args:
        model (VQVAE): The VQ-VAE model instance.
        image_tensor (torch.Tensor): The input image tensor (batch_size, channels, height, width).
        device (torch.device): The device to perform inference on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - reconstructed_image (torch.Tensor): The reconstructed image.
            - perplexity (torch.Tensor): The perplexity of the codebook usage.
            - encoding_indices (torch.Tensor): The discrete encoding indices.
    """
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        reconstructed_image, _, perplexity, encoding_indices = model(image_tensor)
    model.train() # Set the model back to training mode
    return reconstructed_image, perplexity, encoding_indices
