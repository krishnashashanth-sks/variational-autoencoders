import torch.nn as nn
from layers import *
from losses import VQVAELoss

class VQVAE(nn.Module):
  def __init__(self,input_channels:int,num_hiddens:int,num_downsamples:int,num_embeddings:int,embedding_dim:int,commitment_cost:float=0.25):
    super(VQVAE,self).__init__()
    self.encoder=Encoder(input_channels,num_hiddens,num_downsamples)
    self.pre_vq_conv=nn.Conv2d(num_hiddens,embedding_dim,kernel_size=1)
    self.vector_quantizer=VectorQuantizer(num_embeddings,embedding_dim,commitment_cost)
    self.decoder=Decoder(embedding_dim,num_hiddens,num_downsamples)
    self.criterion=VQVAELoss(commitment_cost)
  def forward(self,x:torch.Tensor)->tuple:
    encoder_output=self.encoder(x)
    encoder_output_mapped=self.pre_vq_conv(encoder_output)
    quantized_embeddings_st,_,encoding_indices,perplexity=self.vector_quantizer(encoder_output_mapped)
    reconstructed_x=self.decoder(quantized_embeddings_st)
    loss=self.criterion(reconstructed_x,x,quantized_embeddings_st,encoder_output_mapped)
    return reconstructed_x,loss,perplexity,encoding_indices