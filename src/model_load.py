import torch
import torchvision


class Mynn(torch.nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)
    def forward(self,x):
        x = self.conv1(x)
        return x

#save method1 to load model
model = torch.load("vgg16_method1.pth")
print(model)

#method2 to load model
vgg16 = torchvision.models.vgg16(pretrained=False)
model = torch.load("vgg16_method2.pth")
vgg16.load_state_dict(model)
print(vgg16)


model = torch.load("mynn_method.pth")
print(model)
