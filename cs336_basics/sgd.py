from collections.abc import Callable ,  Iterable
from typing import Optional 
import torch 
import math 
from torch.optim.optimizer import Optimizer
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


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if p not in self.state:
                    self.state[p] = {
                        "step": 0,
                        "m": torch.zeros_like(p),
                        "v": torch.zeros_like(p),
                    }

                state = self.state[p]
                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                #keep the previous 90 percent and take only the new 10percent 
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                state["m"] = m
                state["v"] = v

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                update = lr * m_hat / (v_hat.sqrt() + eps)

                decay = weight_decay * lr * p.data

                p.data = p.data - update - decay

        return loss


import math

def learning_rate_schedule(t, alpha_max, alpha_min, tw, tc):
    if t < tw:
        return (t / tw) * alpha_max
    elif t > tc:
        return alpha_min
    else:
        cosine = math.cos((t - tw) / (tc - tw) * math.pi)
        return alpha_min + 0.5 * (1 + cosine) * (alpha_max - alpha_min)



def gradient_clipping(parameters, max_l2_norm):
    grads =[p.grad for p in parameters if p.grad is not None]
    eps = 1e-6
    total_norm = torch.norm(torch.cat([g.view(-1) for g in grads]), 2)

    if total_norm > max_l2_norm:
        scale  = max_l2_norm/(total_norm+ eps)

        for p in parameters:
            if p.grad is not None:

                p.grad.data.mul_(scale)

    

        
    
