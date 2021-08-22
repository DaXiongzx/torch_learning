import torch
from torch import nn

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        outptut = input+1
        return outptut

Mynn = MyNN()
x = torch.tensor(1.0)
output = Mynn(x) #input:x  then use forward function to get output
print(output)