import math
import torch 
import torch.nn as nn 
import numpy as np 

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float, d_k: int , max_seq_len: int , device=None ):
        super().__init__()
        self.theta = theta 
        self.d_k = d_k 
        self.max_seq_len = max_seq_len
        factory_kwargs = {'device':device}
        # the output of inv_freq would be a 1d matrix of shape (d_k//2,)
        #why is the shape d_k//2 
        # θi,k = i
        # Θ2k/d for k ∈ {1, . . . , d/2} from the paper 
        #k started from 1 but we did 0 
        # but still testcases passed 
        

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        positions = torch.arange(max_seq_len)

        angles =torch.outer(positions, inv_freq)
        sin = torch.zeros((max_seq_len, d_k))
        cos = torch.zeros((max_seq_len,d_k))
        sin[: , 0::2] = torch.sin(angles)
        cos[:, 0::2] = torch.cos(angles)
        sin[:, 1::2] = torch.sin(angles)
        cos[:, 1::2] = torch.cos(angles)


        self.theta_base = 10000 
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def compute_sin_cos(self, max_seq_len , d_model):
        sin_matrix = torch.zeros((max_seq_len, d_model))
        cos_matrix = torch.zeros((max_seq_len, d_model))

        for i in range(max_seq_len):
            for k in range(d_model):
                exponent = 2* (k//2)/d_model 
                theta = i/(self.theta_base**exponent)
                angle = theta * math.pi 
                cos_matrix[i,k] =  math.cos(angle)
                sin_matrix[i,k] = math.sin(angle)   

        return sin_matrix, cos_matrix
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        #x - > (... seq_len , d_k)
        #token pos -> (....seqlen)
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]


        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rotated= torch.zeros_like(x)
        x_rotated[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
        x_rotated[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        return  x_rotated

