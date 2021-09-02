import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.model import *


#准备数据集


train_data = torchvision.datasets.CIFAR10(root="../dataset",train=True,transform=torchvision.transforms.ToTensor(),#PIL->tensor
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size =len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#利用dataloader来加载数据集

train_dataloader = DataLoader(train_data,batch_size=64) #batch_size = 64
test_dataloader = DataLoader(test_data,batch_size=64)

#创建网络模型
mynn = Mynn()
mynn = mynn

#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(mynn.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 30
total_loss = 0

#添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("---------第{}轮训练开始---------".format(i+1))
    #训练开始
    mynn.train()
    for data in train_dataloader:
        imgs,targets = data
        outputs = mynn(imgs)
        #print("outputs:{}",format(outputs))
        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step+1
        print("训练次数：{}，loss{}".format(total_train_step,loss))
        total_loss = total_loss + loss
        writer.add_scalar("train_loss",loss.item(),total_train_step)
    print("total_loss:{}".format(total_loss))
    total_loss = 0
    #测试步骤
    mynn.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = mynn(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss+loss
            accuracy = (outputs.argmax(1)==targets).sum() #64个64个的比较
            total_accuracy = total_accuracy+accuracy
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    print("整体测试集上的loss：{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step+1
    torch.save(mynn,"mynn_{}.pth".format(i))

writer.close()






