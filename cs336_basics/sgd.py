from collections.abc import Callable ,  Iterable
from typing import Optional 
import torch 
import math 

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0 :
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure() 
        # print("self.param_groups " , self.param_groups)
        for group in self.param_groups:
            lr = group["lr"]
            for p in group['params'] :
                if p.grad is None:
                    continue
                state = self.state[p]
                #print("state : ", state)
                t=  state.get("t", 0 )
                grad = p.grad.data 
                p.data -= lr /math.sqrt(t+1) * grad
                state['t'] = t+1 
        return loss 
    