import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernal = torch.tensor([[1,2,1], #卷积层，内核层即权重层
                       [0,1,0],
                       [2,1,0]])
input = torch.reshape(input,(1,1,5,5)) #5*5 batchsize = 1 chanel = 1
kernal = torch.reshape(kernal,(1,1,3,3)) #3*3

output = F.conv2d(input,kernal,stride=1)
print(output)

output2 = F.conv2d(input,kernal,stride=2)
print(output2)

output3 = F.conv2d(input,kernal,stride=1,padding=1)
print(output3)