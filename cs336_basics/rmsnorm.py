import torch 
import torch.nn as nn 
import numpy as np 

class RMSNORM(nn.Module):
    def __init__(self, d_model:int , eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device , 'dtype': dtype}
        self.d_model = d_model 
        self.eps = eps 
        g = torch.ones((d_model, ),  **factory_kwargs)
        self.g = nn.Parameter(g)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        # x-> b seq_len d_model 
        # g -> (d_model,)

        in_dtype = x.dtype 
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps )

        normal_x = x/rms 
        results = normal_x * self.g 
        return results.to(in_dtype)


