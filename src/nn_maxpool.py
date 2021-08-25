import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset",train = False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)




class Mynn(nn.Module):
    def __init__(self):
        super(Mynn,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output

mynn = Mynn()
writer = SummaryWriter("../logs_maxpool")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = mynn(imgs)
    # output = torch.reshape()   maxpool并不改变原来图像的channel 而卷积需要reshape  一层卷积 一层池化
    writer.add_images("output",output,step)
    step = step+1

writer.close()
