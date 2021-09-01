import torchvision
import torch
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

#save model method1 structure+param of the model
torch.save(vgg16,"vgg16_method1.pth")

#save model method2 (official introduce save params)
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#陷阱
class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)
    def forward(self,x):
        x = self.conv1(x)
        return x

mynn = Mynn()
torch.save(mynn,"mynn_method.pth")

