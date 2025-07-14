import torch 
import torch.nn as nn 
import argparse
from cs336_basics.transformerlm import TransformerLM
from cs336_basics.sgd import SGD, AdamW
from cs336_basics.softmax import cross_entropy
from cs336_basics.dataloader import data_loading, load_checkpoint, save_checkpoint
import pickle 
from cs336_basics.bpe_tokenizer import Tokenizer
import numpy as np 
#how to load data
# how to do something with the data
# how to get the target labels 
# we already have a method for data loading 
# 
def main():
    # parser = argparse.ArgumentParser(description="Process some arguments.")
    # parser.add_argument("--vocabsize" , type=int, help= "Vocabsize to use")
    # parser.add_argument("--contextlength", type=int , help="Context length to be used"  )
    # parser.add_argument("--numlayers", type=int, help="Num of layers")
    # parser.add_argument("--dmodel", type=int, help='Dimension to use')
    # parser.add_argument("--numheads", type=int, help="Num of heads")
    # parser.add_argument("--dff", type=int , help="Inner dimensions")
    # parser.add_argument("--outputpath", type=str , help="Provide output path")
    # args = parser.parse_args()

    # print("vocabsize : ", args.vocabsize)
    # print("contextlength: ", args.contextlength)
    # print("numlayers : ", args.numlayers)
    # print("dmodel : ", args.dmodel)
    # print("numheads: ", args.numheads)
    # print("dff : ",args.dff )
    # print("output path ", args.outputpath)
    # vocab_size = args.vocabsize 
    # context_length = args.contextlength
    # num_layers = args.numlayers 
    # d_model = args.dmodel 
    # num_heads =args.numheads
    # d_ff = args.dff 
    # output_path = args.outputpath 

    # if not all([vocab_size, context_length, num_layers, d_model, num_heads, d_ff]):
    #     print("parameters and hyper parameters cannot be empty")
    #     return 
    

    
    with open("data/tokenizers/owt_valid_vocab.pkl", "rb") as f :
            vocab = pickle.load(f)
    with open("data/tokenizers/owt_valid_merges.pkl", "rb") as f :
            merges = pickle.load(f)

    tknzr = Tokenizer(vocab=vocab, merges=merges)
    with open("data/owt_valid.txt", "rb") as f :
        data = f.read().decode("utf-8")
        #print(data[:5000])

        encoded = tknzr.encode(data[:5000])
    
    dataset= np.array(encoded)

    x,y = data_loading(x=dataset, batch_size=64, context_length=512, device="cpu")
    vocab_size = 1000           # just needs to match dataset token range
    context_length = 512        # keep as is
    num_layers = 3             # small Transformer
    d_model = 128               # hidden size per token
    num_heads = 4               # must divide d_model exactly
    d_ff = 512
    model = TransformerLM(vocab_size=vocab_size, context_length=context_length, num_layers=num_layers,  d_model=d_model , num_heads=num_heads, d_ff= d_ff)
    optimizer = AdamW(model.parameters()) 

    for step  in range(100):
        
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()

        loss.backward() 

        optimizer.step() 


        if step%10 == 0 :
            print(f"Step {step}: loss = {loss.item():.4f}")
        #backward 

        #loss pass 


if __name__ == '__main__':
    main()
