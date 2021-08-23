import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataLoader = DataLoader(dataset,batch_size=64)

class Mynn(nn.Module):
    def __init__(self):
        super(Mynn,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
    def forward(self,x):
        x = self.conv1(x)
        return x

mynn = Mynn()
writer = SummaryWriter("nn_conv2dlogs")
step = 0
for data in dataLoader:
    imgs,targets = data #input torch.Size([64, 3, 32, 32]) torch.size=batch.size=64 inchanel=3 rgb=3
    # print(type(imgs)) #torch.Size([64, 6, 30, 30]) outchanel = 6  ->[,3,30,30]
    output = mynn(imgs) #
    writer.add_images("input",imgs,step)
    writer.add_images("output",torch.reshape(output,(-1,3,30,30)),step)
    step = step+1
    # if step==0:
    #     print(imgs.shape) #torch.size=batch.size=64 inchanel=3 rgb=3
    #     print(output.shape) #outchanel = 6
    # step = step+1