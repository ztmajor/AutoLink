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

import setDataset

# Modify Alexnet
class Net(nn.Module):
    def __init__(self, alexnet):
        super(Net, self).__init__()
        self.alexnet = alexnet
        self.fc1 = nn.Linear(1000, 256, bias=True)
        self.fc2 = nn.Linear(256, 42, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.alexnet(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x[0], dim=1)
        return x

def TrainAndSaveModel(path):
    # 加载训练数据
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
        transforms.CenterCrop(32),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    viz = visdom.Visdom()

    trainset = setDataset.LinkBlocks(path, 32, 'train')
    trainloader = DataLoader(trainset, batch_size=5, shuffle=True, num_workers=2)
    # # 展示训练集
    # index = 1
    # for image, label in trainloader:
    #     print(index)
    #     index+=1
    #     viz.images(trainset.denormalize(image), nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(label.numpy()), win='label', opts=dict(title='batch-y'))
    #     time.sleep(1)

    # 在alexnet神经网络结构基础上修改
    alexnet = torch.hub.load('pytorch/vision:v0.5.0', 'alexnet', pretrained=True)
    net = Net(alexnet)
    net.eval()
    print(net)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 训练部分
    for epoch in range(5):  # 训练的数据量为5个epoch，每个epoch为一个循环
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for i, data in enumerate(trainloader, start=0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels

            # # wrap them in Variable
            # inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式用Variable

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = net(inputs)  # 把数据输进CNN net
            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新

            # loss累加
            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用

    print('Finished Training')
    # # 保存神经网络
    # torch.save(model, 'net.pkl')  # 保存整个神经网络的结构和模型参数
    # torch.save(model.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数

# def LoadTestData(path):
#     testset = ImageFolder(path,transform=transforms.Compose([
#                             transforms.Resize((32, 32)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
#                             transforms.ToTensor(),
#                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                           ]))
#     testloader = DataLoader(testset, batch_size=25, shuffle=True, num_workers=2)
#     return testloader
#
#
# def reload_net():
#     trainednet = torch.load('net.pkl')
#     return trainednet
#
#
# def test(path):
#     testloader = LoadTestData(path)
#     net = reload_net()
#     dataiter = iter(testloader)
#     images, labels = dataiter.next()  #
#     imshow(torchvision.utils.make_grid(images, nrow=5))  # nrow是每行显示的图片数量，缺省值为8
#     print('GroundTruth: ' , " ".join('%5s' % classes[labels[j]] for j in range(25)))  # 打印前25个GT（test集里图片的标签）
#     outputs = net(Variable(images))
#     _, predicted = torch.max(outputs.data, 1)
#
#     print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(25)))
#     # 打印前25个预测值


if __name__=='__main__':
    path = ".\\Auxiliary\\DataSet"  # 路径
    TrainAndSaveModel(path)

    # test(path)

    # filename = ".\\dog.jpg"
    # input_image = PIL.Image.open(filename)
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # input_tensor = preprocess(input_image)
    # input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    #
    # # move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')
    #
    # with torch.no_grad():
    #     output = model(input_batch)
    #
    # # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print("print output-------------------------------")
    # print(output[0])
    # # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # # print("softmax------------------------------------")
    # # print(torch.nn.functional.softmax(output[0], dim=0))



