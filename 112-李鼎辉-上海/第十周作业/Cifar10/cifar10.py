import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./cifar10Data',train=True,download=True,
                                        transform=transform)
testset = torchvision.datasets.CIFAR10(root='./cifar10Data',train=True,
                                       download=True,
                                       transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def imshow(img):
    img=img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
dataiter = iter(trainloader)
images, labels = dataiter.__next__()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s'%classes[labels[j]]for j in range(4)))

# 构建卷积神经网络
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # 卷积CNN：input:3,output:16,stride=1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1)
        # 最大池化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积CNN：input:16,output:36,stride:1
        self.conv2 = nn.Conv2d(16, 36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接，Input:1296,output:128
        self.fc1 = nn.Linear(1296, 128)
        # 全连接，Input:128,output:10
        self.fc2 = nn.Linear(128, 10)
    def forward(self,x):
        # conv2d->relu->maxpool
        x = self.pool1(F.relu(self.conv1(x)))
        # conv2d->relu->maxpool
        x= self.pool2(F.relu(self.conv2(x)))
        # x.view()就是将tensor进行reshape,转变成1维
        x = x.view(-1,36*6*6)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x
# 建立神经网络模型
net = CNNNet()
net = net.to(device)

# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# SGD权值优化
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练模型
for epoch in range(10):
    running_los = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs,labels = data
        inputs,labels = inputs.to(device), labels.to(device)
        # 权重参数梯度清0
        optimizer.zero_grad()
#         正向及其反向传播
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        # 显示损失值
        running_los+=loss.item()
        if i%2000 == 1999:
            print('[%d,%5d] loss:%.3f'%(epoch+1, i+1, running_los/2000))
print("finished Training")

class_correct = list(0.for i in range(10))
class_total = list(0.for i in range(10))
with torch.no_grad():
    for data in testloader:
        images,labels = data
        images, labels = images.to(device),labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted==labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label]+=c[i].item()
            class_total[label]+=1
for i in range(10):
    print('Accuracy of %5s : %2d %%' %(classes[i], 100*class_correct[i]/class_total[i]))
