import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
import torch

dataset = torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Mynn(nn.Module):
    def __init__(self):
        super(Mynn,self).__init__()
        self.linear1 = Linear(196608,10) #input_feature and output_feature
    def forward(self,input):
        return self.linear1(input)

mynn = Mynn()

for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = mynn(output)
    print(output.shape)