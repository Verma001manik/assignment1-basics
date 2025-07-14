import torch 
from einops import rearrange, reduce, repeat, einsum 
import numpy as np 
from tinygrad import Tensor
import torch.nn as nn 
import math
import numpy as np
from cs336_basics.sgd import SGD
from collections.abc import Callable ,  Iterable
from typing import Optional 

import torch

param = torch.tensor([[0.5, -0.3], [0.1, 0.8]], requires_grad=True)

lr = 0.1        
beta1 = 0.9        
beta2 = 0.999      
eps = 1e-8       
lam = 0.01        
m = torch.zeros_like(param)
v = torch.zeros_like(param)
t = 1  

grad = param.grad.data

m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad * grad

lr_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))

param.data = param.data - lr_t * m / (v.sqrt() + eps)
