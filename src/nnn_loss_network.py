import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1)

class Mynn(nn.Module):
    def __init__(self): #定义了网络的一系列操作
        super(Mynn,self).__init__()
        self.conv1 = Conv2d(3,32,5,padding=2) #in_channel out_channel kernel_size
        self.maxpool1 = MaxPool2d(2)
        self.conv2 =Conv2d(32,32,5,padding=2) #维持尺寸不变，padding=2
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32,64,5,padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024,64) #in_feature out_feature
        self.linear2 = Linear(64,10)
        self.module1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),  # 维持尺寸不变，padding=2
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x = self.module1(x)
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        return x

loss = nn.CrossEntropyLoss()
mynn = Mynn()
for data in dataloader:
    imgs,targets = data

    # print(len(imgs))
    # print(targets)
    output = mynn(imgs)
    result_loss = loss(output,targets)
    result_loss.backward()
    print("ok")


