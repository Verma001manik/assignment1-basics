import torch 
from einops import rearrange, reduce, repeat, einsum 
import numpy as np 
from tinygrad import Tensor
import torch.nn as nn 
import math
import numpy as np
n_embeddings = torch.randn((3,6,64))
#print(n_embeddings)
# max_seq_len = 6
# d_model = 64
# theta = 10000.0
# pos_ids = torch.tensor([ 0,  1,  2,  3,  4,  5])

# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(self,theta:float, d_k: int , max_seq_len: int , device=None ):
#         super().__init__()
#         self.theta = theta 
#         self.d_k = d_k 
#         self.max_seq_len = max_seq_len
#         factory_kwargs = {'device':device}
#         inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
#         #print("inv_freq : ", inv_freq)
#         #print("shape: \n", inv_freq.shape)
#         positions = torch.arange(max_seq_len)

#         angles =torch.outer(positions, inv_freq)
#         #print(angles)
        
#         sin = torch.zeros((max_seq_len, d_k))
#         cos = torch.zeros((max_seq_len,d_k))
#         #print(sin)

#         sin[: , 0::2] = torch.sin(angles)
#         #print(sin)
#         cos[:, 0::2] = torch.cos(angles)
#         sin[:, 1::2] = torch.sin(angles)
#         print(sin)
#         cos[:, 1::2] = torch.cos(angles)


#         self.theta_base = 10000 
#         self.register_buffer("sin", sin, persistent=False)
#         self.register_buffer("cos", cos, persistent=False)

   
#     def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
#         #x - > (... seq_len , d_k)
#         #token pos -> (....seqlen)
#         sin = self.sin[token_positions]
#         cos = self.cos[token_positions]

#         print('sin also : ', sin.shape)
#         x1 = x[..., 0::2]
#         x2 = x[..., 1::2]
#         x_rotated= torch.zeros_like(x)
#         x_rotated[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
#         x_rotated[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
#         return  x_rotated



# t = RotaryPositionalEmbedding(theta, d_model, max_seq_len)
# t(n_embeddings, pos_ids) 
seq_len = torch.zeros((3,3))
seq_len[:, 0::2]= 1
a = torch.randn((3,3))
print(a)

a = a.masked_fill(seq_len==0, float('-inf'))
print(a)

# ok so to change a , the desision is in control of mask and not a 
# wherever we see true in mask 
# there we do 