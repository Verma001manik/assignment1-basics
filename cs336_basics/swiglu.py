import torch 
import torch.nn as nn
import numpy as np 
from cs336_basics.linear import Linear 

# dont need a silu class man 
class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x ):
        out = x * torch.sigmoid(x)
        return out 
    

class GLU(nn.Module):
    def __init__(self, in_dim , out_dim):
        super().__init__()
        self.w1 = Linear(in_dim , out_dim)
        self.w2 = Linear(in_dim, out_dim)


    
    def forward(self, x ):
        left = torch.sigmoid(self.w1(x))
        right = self.w2(x)
        out = left * right 

        return out 
    

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)  
        self.w2 = Linear(d_ff, d_model)  
        self.w3 = Linear(d_model, d_ff)  
        

    def forward(self, x):
        x1 = self.w1(x)                    
        x3 = self.w3(x)                     
        gated = torch.sigmoid(x1) * x1      
        out = self.w2(gated * x3)           #
        return out
