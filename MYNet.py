import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import visdom
import time
from PIL import Image
import cv2
import os

import setDataset

# Modify Alexnet
class Net(nn.Module):
    def __init__(self, alexnet):
        super(Net, self).__init__()
        self.alexnet = alexnet
        self.fc1 = nn.Linear(1000, 256, bias=True)
        self.fc2 = nn.Linear(256, 42, bias=True)
        self.bn1 = nn.BatchNorm1d(42)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.alexnet(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.bn1(x)
        x = self.softmax(x)
        return x

# 训练集
def TrainAndSaveModel(path):
    # # visdom对象
    # viz = visdom.Visdom()

    # 加载训练数据
    trainset = setDataset.LinkBlocks(path, 64, 'train')
    trainloader = DataLoader(trainset, batch_size=20, shuffle=True, num_workers=2)
    # # 展示训练集
    # index = 1
    # for image, label in trainloader:
    #     print(index)
    #     index+=1
    #     viz.images(trainset.denormalize(image), nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(label.numpy()), win='label', opts=dict(title='batch-y'))

    # 在alexnet神经网络结构基础上修改
    alexnet = torch.hub.load('pytorch/vision:v0.5.0', 'alexnet', pretrained=True)
    net = Net(alexnet)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.is_available())
    net = net.to(device)
    # print(net)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 训练部分
    for epoch in range(10):  # 训练的数据量为1000个epoch，每个epoch为一个循环
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for i, data in enumerate(trainloader, start=0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = net(inputs)  # 把数据输进CNN net
            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新

            # print("i=", i)
            # loss累加
            running_loss += loss.item()
            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))  # 然后再除以20，就得到这二十次的平均损失值
                running_loss = 0.0  # 这一个20次结束后，就把running_loss归零，下一个30次继续使用

    print('Finished Training')
    # 保存整个神经网络
    torch.save(net, 'alexnet.pth')
    # 保存参数
    torch.save(net.state_dict(), 'alexnet_state.pth')
    print('Saving Completed')

# 测试集
def LoadAndTestData(path):
    # visdom对象
    viz = visdom.Visdom()

    testset = setDataset.LinkBlocks(path, 64, 'test')
    testloader = DataLoader(testset, batch_size=8, shuffle=True, num_workers=2)

    net = torch.load('alexnet.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # # 展示测试集
    # index = 1
    # for image, label in testloader:
    #     print(index)
    #     index+=1
    #     viz.images(testset.denormalize(image), nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(label.numpy()), win='label', opts=dict(title='batch-y'))
    #     time.sleep(1)


    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    # images, labels = images.to(device), labels.to(device)
    #
    # viz.images(testset.denormalize(images), nrow=8, win='batch', opts=dict(title='batch'))
    # viz.text(str(labels.numpy()), win='label', opts=dict(title='batch-y'))
    #
    # outputs = net(images)
    # _, predicted = torch.max(outputs.data, 1)
    # print(predicted)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))


if __name__=='__main__':
    t = time.time()
    path = ".\\Auxiliary\\DataSet"  # 路径
    # # 训练模型，并保存
    # TrainAndSaveModel(path)

    # # 读取模型，并测试
    # LoadAndTestData(path)

    device = torch.device('cuda')
    transform = transforms.Compose([
        transforms.Resize(64),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testnet = torch.load('alexnet.pth')
    testnet = testnet.to(device)
    testnet.eval()
    torch.no_grad()
    img = Image.open('./IMG/test/(1,1)(1).jpg')
    img = transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = testnet(img_)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    classes = ('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19',
               '20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39',
               '40','41')
    print('this picture maybe :', classes[predicted[0]])

    t1 = time.time()
    print("用时：", t1 - t)
