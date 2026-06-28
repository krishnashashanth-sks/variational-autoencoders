import torch
import torch.nn as nn
class Decoder(nn.Module):
  def __init__(self,input_channels,num_hiddens,num_upsamples):
    super(Decoder,self).__init__()
    self.layers=nn.ModuleList()
    self.layers.append(nn.Sequential(
        nn.Conv2d(input_channels,num_hiddens//(2**(num_upsamples-1)),kernel_size=3,padding=1),
        nn.ReLU()
    ))
    current_channels=num_hiddens//(2**(num_upsamples-1))
    for i in range(num_upsamples-1):
      next_channels=current_channels
      self.layers.append(nn.Sequential(
          nn.ConvTranspose2d(current_channels,next_channels,kernel_size=4,stride=2,padding=1),
          nn.ReLU()
      ))
      current_channels=next_channels
    self.layers.append(nn.Sequential(
        nn.ConvTranspose2d(current_channels,3,kernel_size=4,stride=2,padding=1),
        nn.Tanh()
    ))
  def forward(self,x):
    for layer in self.layers:
      x=layer(x)
    return x
  
import torch
import torch.nn as nn
import torch.nn.functional as F
class VectorQuantizer(nn.Module):
  def __init__(self,num_embeddings,embedding_dim,commitment_cost:float):
    super(VectorQuantizer,self).__init__()
    self.num_embeddings=num_embeddings
    self.embedding_dim=embedding_dim
    self.commitment_cost=commitment_cost
    self.embedding=nn.Embedding(self.num_embeddings,self.embedding_dim)
    self.embedding.weight.data.uniform_(-1/self.num_embeddings,1/self.num_embeddings)
  def forward(self,enc_output:torch.Tensor)->tuple:
    original_shape=enc_output.shape
    flat_input=enc_output.permute(0,2,3,1).contiguous()
    flat_input=flat_input.view(-1,self.embedding_dim)
    distances=(torch.sum(flat_input**2,dim=1,keepdim=True)+torch.sum(self.embedding.weight**2,dim=1)-2*torch.matmul(flat_input,self.embedding.weight.T))
    encoding_indices=torch.argmin(distances,dim=1).unsqueeze(1)
    quantized_embeddings=self.embedding(encoding_indices).view(original_shape)
    quantized_embeddings_st=enc_output+(quantized_embeddings-enc_output).detach()
    encodings=F.one_hot(encoding_indices.squeeze(1),self.num_embeddings).float()
    avg_probs=torch.mean(encodings,dim=0)
    perplexity=torch.exp(-torch.sum(avg_probs*torch.log(avg_probs+1e-10)))
    return quantized_embeddings_st,enc_output,encoding_indices.view(original_shape[0],original_shape[2],original_shape[3]),perplexity

class Encoder(nn.Module):
  def __init__(self,input_channels:int,num_hiddens:int,num_downsamples:int):
    super(Encoder,self).__init__()
    self.layers=nn.ModuleList()
    current_channels=input_channels
    self.layers.append(nn.Sequential(
        nn.Conv2d(current_channels,num_hiddens//(2**num_downsamples),kernel_size=4,stride=2,padding=1),
        nn.ReLU()
    ))
    current_channels=num_hiddens//(2**num_downsamples)
    for i in range(num_downsamples-1):
      next_channels=current_channels*2
      self.layers.append(nn.Sequential(
          nn.Conv2d(current_channels,next_channels,kernel_size=4,stride=2,padding=1),
          nn.ReLU()
      ))
      current_channels=next_channels
    self.layers.append(nn.Sequential(
        nn.Conv2d(current_channels,num_hiddens,kernel_size=3,stride=1,padding=1)
    ))
  def forward(self,x):
    for layer in self.layers:
      x=layer(x)
    return x