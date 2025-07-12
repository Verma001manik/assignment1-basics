import torch
import torch.nn as nn
from cs336_basics.mulithead import CausalMultiHeadAttention
from cs336_basics.rmsnorm import RMSNORM
from cs336_basics.swiglu import SwiGLU
from jaxtyping import Float
from torch import Tensor

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len):
        super().__init__()  
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        self.rms1 = RMSNORM(d_model=self.d_model)
        self.rms2 = RMSNORM(d_model=self.d_model)
        self.mha = CausalMultiHeadAttention(
            d_model=self.d_model, 
            num_heads=self.num_heads, 
            theta=self.theta, 
            max_seq_len=self.max_seq_len
        )
        self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff)
    
    def forward(self, x, weights):
        batch_size, seq_len, _ = x.shape
        device = x.device
        token_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Extract weights
        q_proj_weight = weights['attn.q_proj.weight']
        k_proj_weight = weights['attn.k_proj.weight']
        v_proj_weight = weights['attn.v_proj.weight']
        o_proj_weight = weights['attn.output_proj.weight']
        ln1_weight = weights['ln1.weight']
        ln2_weight = weights['ln2.weight']
        
        residual = x
        self.rms1.g.data = ln1_weight
        x = self.rms1(x)
        x = self.mha(x, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, token_positions)
        x = x + residual
        
        residual = x
        self.rms2.g.data = ln2_weight
        x = self.rms2(x)
        
        self.ffn.load_state_dict({
            "w1.weight": weights["ffn.w1.weight"],
            "w2.weight": weights["ffn.w2.weight"],
            "w3.weight": weights["ffn.w3.weight"],
        })
        
        x = self.ffn(x)
        x = x + residual
        
        return x
