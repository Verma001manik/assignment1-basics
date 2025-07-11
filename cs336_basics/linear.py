import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device':device , 'dtype':dtype}

        W = torch.empty((out_features, in_features) ,**factory_kwargs)
        nn.init.trunc_normal_(W, std=0.2)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.W.T
        return x 
