import torch.nn as nn 
import torch 
import torch.nn.functional as F 

class Embedding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # num_embeddings -> size of vocab (vocab_size)
        # embedding_dim -> dmodel 
        factory_kwargs = {'device':device , 'dtype':dtype}
        embedding_matrix = torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        nn.init.trunc_normal_(embedding_matrix, mean=0.0, std=1.0)
        self.embedding_table = nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.embedding_table[token_ids]

e = Embedding(3,4)
