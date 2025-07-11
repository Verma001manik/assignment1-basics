import torch 
import torch.nn as nn 
import numpy as np 

def softmax(x,i):
    maxi = torch.max(x, dim=i , keepdim=True).values
    x = x-maxi 

    exp_num = np.exp(x)

    out = exp_num/torch.sum(exp_num, dim=i, keepdim=True)
    return out 



