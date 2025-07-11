import torch 
import torch.nn as nn 

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        #Folllowing Vaswani et al. [2017], set dk = dv = dmodel/h
        h = 8 
        self.d_k = self.d_h = d_model//h 


        self.wo = nn.Linear(self.dv,self.d_model)


        # scaled dot attention will give us the output 
        # scores = b , ..  seq_len , d_v 
        # scores @ wo 
        # b , ... ,seq_len , d_v  @ (d_v , d_model)
        # b , ... , seq_len , d_model 

        self.w_q = nn.Linear(self.d_model , self.d_k)
        self.w_k = nn.Linear(self.d_model , self.d_k)
        self.w_v =nn.Linear(self.d_model, self.d_v)
        



