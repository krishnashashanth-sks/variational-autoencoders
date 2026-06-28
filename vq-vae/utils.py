import numpy as np

def denormalize_img(tensor_img):
    # Assumes normalization to [-1, 1]
    tensor_img = tensor_img * 0.5 + 0.5  # Scale back to [0, 1]
    np_img = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(np_img, 0, 1) # Clip to ensure valid pixel values
