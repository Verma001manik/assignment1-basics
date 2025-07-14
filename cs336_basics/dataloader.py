import numpy as np 
import torch 

def data_loading(x, batch_size, context_length, device: str):
    # what should i return ?
    # pair of tensors , -> tensor 1 , tensor2 
    # both tensors should have shape (batch_size, context_length)
    # x is the dataset 
    # we only sample from the dataset 
    # that means the tensors must contain only those ids that are present in the dataset 
    # 1 2 4 5 6 7 8 9 10 
    # bs = 1 #m = 3 
     #1 row 3 columns for the 2 tensors 
     #for any bs or contextlen we have to return only 2 tensors 
    dataset = torch.from_numpy(x).long()   
    start_indices = np.random.randint(0, len(x) - context_length, size=batch_size) 

    tensor1 = torch.stack([dataset[i: i+context_length] for i in start_indices]) 
    tensor2 = torch.stack([dataset[i+1: i+1+context_length] for i in start_indices])

    x = torch.tensor(tensor1, dtype=torch.long, device=device)
    y = torch.tensor(tensor2, dtype=torch.long, device=device)

    return x,y



def save_checkpoint(model , optimizer, iteration, out):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)



def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = checkpoint["iteration"]

    return iteration