import torch 
from einops import rearrange, reduce, repeat, einsum 
# import numpy as np 
# from tinygrad import Tensor
import torch.nn as nn 
# import math
import numpy as np
# from cs336_basics.sgd import SGD
# from collections.abc import Callable ,  Iterable
# from typing import Optional 



# Simple neural network with 1 hidden layer

# Simple neural network with 1 hidden layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 4)  # input: 3 features, hidden: 4 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)  # output: 1 value

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model instance
model = SimpleNN()

# Print state_dict
# print("State dict:\n")
# for name, param in model.state_dict().items():
#     print(f"{name}:\n{param}\n")
print(model.state_dict())