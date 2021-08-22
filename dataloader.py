import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
#测试数据集中的第一张图片及target
img,target = test_set[0] #__getitem__
print(img.shape)
print(target)
# print(len(test_set))
writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs,targets = data
    writer.add_images("test_dataloader",imgs,step)
    step = step+1
writer.close()