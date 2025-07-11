import torch 
import torch.nn as nn 
from cs336_basics.softmax import softmax
from einops import rearrange , reduce 
def scaled_dot_product_attention(q,k,v, mask=None):
    d_k = q.shape[-1]
    kt = rearrange(k, 'b ... seq_len d_k -> b ... d_k seq_len')
    scores  = q@ kt
    scale = torch.sqrt(torch.tensor(d_k, dtype=q.dtype, device=q.device))
    scores = scores/scale 
    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf')) 
    atten_weights = softmax(scores, dim=-1)

    out =atten_weights@ v 
    return out 


    # q -> b , ... , seq_len, d_k 
    # k = b, ..., seq_len, d_k 

    # kt= b, ..., d_k, seq_len
    # q @ kt --> b,seq_lem, dk --> b, dk,seq_len --> b ,seq_len, seq_len 
    #above @ k -> b , seq_len ,seq_len --> b, seq_len, d_v--> b , seq_len ,d_v 
    #v = b, ..., seq_len , d_v 


    
    


