import torch 
import torch.nn as nn 
import numpy as np 

def softmax(x,dim):
    maxi = torch.max(x, dim=dim , keepdim=True).values
    x = x-maxi 

    exp_num = torch.exp(x)

    out = exp_num/torch.sum(exp_num, dim=dim, keepdim=True)
    return out 



