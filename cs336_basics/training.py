import torch 
import torch.nn as nn 
import argparse
from cs336_basics.transformerlm import TransformerLM
from cs336_basics.sgd import SGD
from cs336_basics.softmax import cross_entropy
def main():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--vocabsize" , type=int, help= "Vocabsize to use")
    parser.add_argument("--contextlength", type=int , help="Context length to be used"  )
    parser.add_argument("--numlayers", type=int, help="Num of layers")
    parser.add_argument("--dmodel", type=int, help='Dimension to use')
    parser.add_argument("--numheads", type=int, help="Num of heads")
    parser.add_argument("--dff", type=int , help="Inner dimensions")
    parser.add_argument("--outputpath", type=str , help="Provide output path")
    args = parser.parse_args()

    print("vocabsize : ", args.vocabsize)
    print("contextlength: ", args.contextlength)
    print("numlayers : ", args.numlayers)
    print("dmodel : ", args.dmodel)
    print("numheads: ", args.numheads)
    print("dff : ",args.dff )
    print("output path ", args.outputpath)
    vocab_size = args.vocabsize 
    context_length = args.contextlength
    num_layers = args.numlayers 
    d_model = args.dmodel 
    num_heads =args.numheads
    d_ff = args.dff 
    output_path = args.outputpath 


    model = TransformerLM(vocab_size=vocab_size, context_length=context_length, num_layers=num_layers,  d_model=d_model , num_heads=num_heads, d_ff= d_ff)
    optimizer = SGD(model.parameters()) 

    for _ in range(100):
        #forward 
        pass 
    
        #backward 

        #loss pass 


if __name__ == '__main__':
    main()
