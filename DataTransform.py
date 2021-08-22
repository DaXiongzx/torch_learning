import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    ]) #[]defines a serval ops 定义了一系列组合操作
train_set = torchvision.datasets.CIFAR10(root = "./dataset",train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)


# print(test_set.classes)
# img,target = test_set[0]
# img.show()
# print(target)
# print(test_set[0])
writer = SummaryWriter("p10logs")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()