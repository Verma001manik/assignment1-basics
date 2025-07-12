import torch 
import torch.nn as nn 
from cs336_basics.tblock import TransformerBlock
from cs336_basics.rmsnorm import RMSNORM
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear 
from cs336_basics.softmax import softmax 
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model ,num_heads, d_ff, theta):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_ff = d_ff 
        self.theta = theta 

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.theta, self.context_length, layer_idx=i)
            for i in range(self.num_layers)
        ])
        self.embedding = Embedding(self.vocab_size, self.d_model)
        self.lm_head = Linear(self.d_model , self.vocab_size)
        self.ln_final = RMSNORM(d_model=self.d_model)
    def forward(self,x, weights ):
        lm_head_weight = weights['lm_head.weight']
        lm_final_weight = weights['ln_final.weight']
        embed_weight = weights['token_embeddings.weight']
        self.embedding.embedding_table.data = embed_weight

        self.ln_final.g.data = lm_final_weight
        x= self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x, weights)
        x = self.ln_final(x)

        self.lm_head.weight.data = lm_head_weight
        logits = self.lm_head(x)
        # out = softmax(logits, dim=-1)

        

        return logits