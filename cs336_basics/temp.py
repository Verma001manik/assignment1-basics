import torch 
from einops import rearrange, reduce, repeat, einsum 
# import numpy as np 
# from tinygrad import Tensor
import torch.nn as nn 
# import math
import numpy as np
# from cs336_basics.sgd import SGD
#
from cs336_basics.dataloader import data_loading
import pickle 

from cs336_basics.bpe_tokenizer import Tokenizer, from_files

with open("data/tokenizers/owt_valid_vocab.pkl", "rb") as f :
        vocab = pickle.load(f)
with open("data/tokenizers/owt_valid_merges.pkl", "rb") as f :
        merges = pickle.load(f)

tknzr = Tokenizer(vocab=vocab, merges=merges)

with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f :
        data = f.read().decode("utf-8")
        #print(data[:5000])

        encoded = tknzr.encode(data[:5000])
        

dataset = np.array(encoded)
x,y = data_loading(dataset, 64, 256, device="cpu")
print(x.shape)  

print(y)