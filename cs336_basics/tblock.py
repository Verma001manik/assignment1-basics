from einops import repeat
import torch
import torch.nn as nn
from cs336_basics.mulithead import CausalMultiHeadAttention
from cs336_basics.rmsnorm import RMSNORM
from cs336_basics.swiglu import SwiGLU
from jaxtyping import Float
from torch import Tensor

# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len,layer_idx):
#         super().__init__()  
#         self.layer_idx = layer_idx
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_ff = d_ff
#         self.theta = theta
#         self.max_seq_len = max_seq_len
        
#         self.rms1 = RMSNORM(d_model=self.d_model)
#         self.rms2 = RMSNORM(d_model=self.d_model)
#         self.mha = CausalMultiHeadAttention(
#             d_model=self.d_model, 
#             num_heads=self.num_heads, 
#             theta=self.theta, 
#             max_seq_len=self.max_seq_len
#         )
#         self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff)
    
#     def forward(self, x, weights):
#         batch_size, seq_len, _ = x.shape
#         device = x.device
#         token_positions = repeat(torch.arange(seq_len, device=device), 's -> b s', b=batch_size)        
#         # Extract weights
#         layer_prefix = f"layers.{self.layer_idx}"
#         q_proj_weight = weights[f"{layer_prefix}.attn.q_proj.weight"]
#         k_proj_weight = weights[f"{layer_prefix}.attn.k_proj.weight"]
#         v_proj_weight = weights[f"{layer_prefix}.attn.v_proj.weight"]
#         o_proj_weight = weights[f"{layer_prefix}.attn.output_proj.weight"]

#         layer_prefix = f"layers.{self.layer_idx}"
#         ln1_weight = weights[f"{layer_prefix}.ln1.weight"]
#         ln2_weight = weights[f"{layer_prefix}.ln2.weight"]

        
#         residual = x
#         self.rms1.g.data = ln1_weight
#         x = self.rms1(x)
#         x = self.mha(x, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, token_positions)
#         x = x + residual
        
#         residual = x
#         self.rms2.g.data = ln2_weight
#         x = self.rms2(x)
        
#         self.ffn.load_state_dict({
#             "w1.weight": weights[f"{layer_prefix}.ffn.w1.weight"],
#             "w2.weight": weights[f"{layer_prefix}.ffn.w2.weight"],
#             "w3.weight": weights[f"{layer_prefix}.ffn.w3.weight"],
#         })

        
#         x = self.ffn(x)
#         x = x + residual
        
#         return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
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

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        token_positions = repeat(torch.arange(seq_len, device=device), 's -> b s', b=batch_size)

        residual = x
        x = self.rms1(x)
        x = self.mha(x, token_positions)
        x = x + residual

        residual = x
        x = self.rms2(x)
        x = self.ffn(x)
        x = x + residual

        return x