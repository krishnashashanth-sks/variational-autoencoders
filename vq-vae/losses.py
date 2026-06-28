import torch
import torch.nn as nn
class VQVAELoss(nn.Module):
  def __init__(self,commitment_cost:float=0.25):
    super(VQVAELoss,self).__init__()
    self.mse_loss=nn.MSELoss()
    self.commitment_cost=commitment_cost
  def forward(self,reconstructed_x:torch.Tensor,x:torch.Tensor,quantized_embeddings:torch.Tensor,encoder_output:torch.Tensor)->torch.Tensor:
    reconstruction_loss=self.mse_loss(reconstructed_x,x)
    quantization_loss=self.mse_loss(quantized_embeddings.detach(),encoder_output)
    commitment_loss=self.mse_loss(encoder_output,quantized_embeddings.detach())*self.commitment_cost
    total_loss=reconstruction_loss+quantization_loss+commitment_loss
    return total_loss