# import torch 
# import torch.nn as nn 
# from cs336_basics.rope import RotaryPositionalEmbedding
# from cs336_basics.scaled import scaled_dot_product_attention
# from einops import rearrange
# class CausalMultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super().__init__()

#         self.d_model = d_model
#         self.num_heads = num_heads
#         #Folllowing Vaswani et al. [2017], set dk = dv = dmodel/h
#         h = 8 
#         self.d_k = self.d_v = d_model//self.num_heads 


#         self.wo = nn.Linear(self.d_v * self.num_heads, self.d_model)



#         # scaled dot attention will give us the output 
#         # scores = b , ..  seq_len , d_v 
#         # scores @ wo 
#         # b , ... ,seq_len , d_v  @ (d_v , d_model)
#         # b , ... , seq_len , d_model 

#         self.w_q = nn.Linear(self.d_model , self.d_k*num_heads)
#         self.w_k = nn.Linear(self.d_model , self.d_k*num_heads)
#         self.w_v =nn.Linear(self.d_model, self.d_v*num_heads)

#         # d_in means d_model 
        

    
#     def forward(self, x):
#         # "x =  ... sequence_length d_in"
#         # w_k = d_in, d_in
#         # w_k(x) --> seq_len d_in @ d_in  d_in -> seq_len d_in
#         #w_q = d_in , d_k 
#         #w_q(x) = seq_len d_in @ d_in d_in -> seq_len d_in 


#         # change each q/k/v into seq_len d_k/d_v

#         # b , seq_len 512 -> b ,seq_len 8 64 -> b 8 seq_len 64 
#         q = self.w_q(x)
#         k = self.w_k(x)
#         v = self.w_v(x)

#         q = rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k",h=self.num_heads)
#         k = rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k",h=self.num_heads)
#         v  = rearrange(v, "... seq_len (h d_v) -> ... h seq_len d_v",h=self.num_heads)
#         attn_scores = scaled_dot_product_attention(q,k,v)
        
#         out = rearrange(attn_scores, "... h seq_len d_v -> ... seq_len (h d_v)")
#         # convert each of them back into seq_len, d_in
#         out = self.wo(out)
#         #wo -> d_model, d_model 
#         return out 


import torch 
import torch.nn as nn 
from einops import rearrange
from cs336_basics.scaled import scaled_dot_product_attention 

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // self.num_heads

    def forward(
        self,
        x,                    # shape: (..., seq_len, d_model)
        q_proj_weight,        # shape: (d_k * num_heads, d_model)
        k_proj_weight,        # shape: (d_k * num_heads, d_model)
        v_proj_weight,        # shape: (d_v * num_heads, d_model)
        o_proj_weight         # shape: (d_model, d_v * num_heads)
    ):
        
        seq_len = x.shape[-2]
        device = x.device
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)
        q = x @ q_proj_weight.T  # (..., seq_len, d_k * num_heads)
        k = x @ k_proj_weight.T
        v = x @ v_proj_weight.T

        q = rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        k = rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        v = rearrange(v, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)

        out = scaled_dot_product_attention(q, k, v, mask=mask)  # (..., h, seq_len, d_v)

        out = rearrange(out, "... h seq_len d_v -> ... seq_len (h d_v)")

        out = out @ o_proj_weight.T  # (..., seq_len, d_model)

        return out
