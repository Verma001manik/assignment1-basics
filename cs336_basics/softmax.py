import torch 
import torch.nn as nn 
import numpy as np 

def softmax(x,dim):
    maxi = torch.max(x, dim=dim , keepdim=True).values
    x_stable  = x-maxi 
    exp_x = torch.exp(x_stable)
    sum_exp_x = exp_x.sum(dim=dim,keepdim=True)

    out = exp_x/sum_exp_x
    return out 
def log_softmax(x, dim):
    x_off = x - torch.max(x, dim=dim, keepdim=True).values
    return x_off - torch.log(torch.sum(torch.exp(x_off), dim=dim, keepdim=True))  

def cross_entropy(logits, targets):
    log_probs = log_softmax(logits, dim=-1)    
    batch = logits.shape[0]
   
    selected_log_probs = log_probs[torch.arange(batch), targets]
   
    # print("Selected log probabilities:", selected_log_probs)
   
    loss = -selected_log_probs.mean()  
    return loss