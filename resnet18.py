import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pylab as plt
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2
import gradio as gr
import os

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'训练集转换'
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),#随机翻转
    transforms.RandomCrop(32,padding=4),#剪裁
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
 
'测试集转换'
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
train_data = torchvision.datasets.CIFAR10(root='./cifar10', train=True, transform=train_transforms,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./cifar10', train=False, transform=test_transforms,
                                         download=True)
train_dataloader = DataLoader(train_data, batch_size=128,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128,shuffle=True)

class ResidualBlock(nn.Module):
    def __init__(self, conv, bn, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv # 就是一个卷积层，只不过由外部作为参数传入
        self.bn1 = bn# 归一化，就是将里面的每一个值映射为0->1区间，为了防止过拟合，过拟合是啥呢，我也不太清楚，但是反应的结果就是训练时的损失值很小，但是测试时的损失值很大
        self.relu = nn.ReLU(inplace=True)# 激活函数，也是一个映射，具体百度或者看我卷积神经网络的博客
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample# 外部会传入此参数，决定是否进行A1+B1作为C层输入的操作
        self.stride = stride # 卷积核大小

    def forward(self, x):
        identity = x  # 这里就相当于先吧A1存储，如果后续需要用到就使用
        out = self.conv1(x) # 下面4步骤依次为：卷积、归一、激活、卷积、归一
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 然后判断需要需要进行A1+B1作为C层输入的操作、如果需要，因为x的值与最新的out维度大小啥的是不匹配的，需要加一些操作使其匹配
        # 因为要将out向量与identity相加求和再传递给下一个残差块，但是out是经过卷积了，且第一次卷积，卷积核数为2，
        # 即，经过第一次卷积后特征图大小减半，但由于全部特征图应该保持不变，所以我们将输入通道数由64变为128，
        # 也因此、为了identity与out匹配，即也需要将identity从64的通道数变为128，故加此一层
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity # 相当于 A1+B1
        out = self.relu(out) # 激活

        return out # 作为返回值，这样下一个残差块获得的输入就是 A1+B1 了

class resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super(resnet18, self).__init__()
        # 具体方法参数在卷积神经网络博客中有详细介绍，我这里只说为啥是这个值
        # 因为图片是三通道，输出通道是64、然后将卷积核设置为7X7(它这里是简写)，卷积核步长:2，将图片四周多加3块，变成35X35，这样更好提取边缘特征
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)  # 归一化 维度不变，变的只有里面的值
        self.relu = nn.ReLU(inplace=True)
        # 池化
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # 64：通道数，2：残差块数量、blocks：只循环执行次数，为2时执行一次
        self.layer1 = self._make_layer(planes=64, blocks=2)
        self.layer2 = self._make_layer(planes=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None # 先初始化
        # 大致意思是，我们将第一个残差块不进行 A1+B1的操作，将后续的残差块进行，为啥我也不知道
        if stride != 1 or planes != 64:
            # 此处就是为 A1能够与B1相加做准备，将A1进行卷积与归一后其维度才与B1相同
            downsample = nn.Sequential(
                nn.Conv2d(int(planes/stride), planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
		# 用来存储一个  self.layer1(x) 方法中需要多少个残差块，这由不同深度的残差神经网络去决定，这里是两个，固定一个加上通过变量blocks的值去决定是否还加，这样写也就是为了扩展性，不这样写也行
        layers = []
        '''
                 参数为啥要这样写 nn.Conv2d(int(planes/stride), planes, kernel_size=3, stride=stride
         正常情况下，如果stride=1，那么通道数应该是输入多少输出就是多少，但由于stride有等于2的情况，所以我们在初始通道数需要进行除法，但是除法后值是浮点数，而参数需要整型，所以使用int()，而且我这里这样写是为了迎合：
         self.layer1 = self._make_layer(planes=64, blocks=2)
         即我们开始定义的输入，因为planes变量是作为输出的定义，所以我们需要计算输入值、当输入变成：
         self.layer2 = self._make_layer(planes=128, blocks=2, stride=2)
         时，为了保证输入是输出的一半，所以这样写、也可以自己改
        '''
        layers.append(ResidualBlock(nn.Conv2d(int(planes/stride), planes, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(planes), planes, 1, downsample))
        for i in range(1, blocks):
            layers.append(ResidualBlock(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(planes), planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # # 全局平均池化层、它的作用是将每个特征图上的所有元素取平均，得到一个指定大小的输出。在 ResNet18 中，该池化层的输出大小为 。(batch_size, 512, 1, 1)用于最终的分类任务
        x = self.avgpool(x)
        # # 作用是将张量  沿着第 1 个维度进行压平，即将  转换为一个 1 维张量，
        x = torch.flatten(x, 1)
        # # 这里对应模型的 self.fc = nn.Linear(512, num_classes)，就是将一维向量经过映射缩小到 10，因为CIFAR10是个10分类问题
        x = self.fc(x)

        return x
        
Epoch=1
device = torch.device('cuda')
model=resnet18().to(device)
# print(model)
# os.system("pause")
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.CrossEntropyLoss()
accuracies = []

# def train(Epoch=1):
#     accuracies=[]
#     Epoch=int(Epoch)
#     for epoch in range(Epoch):
#     #step,代表现在第几个batch_size
#     #batch_x 训练集的图像
#     #batch_y 训练集的标签
#         for step, (batch_x, batch_y) in enumerate(train_dataloader):
#             #model只接受Variable的数据，因此需要转化
#             b_x=Variable(batch_x)
#             b_x=b_x.to(device)
#             b_y=Variable(batch_y)
#             b_y=b_y.to(device)
#             #将b_x输入到model得到返回值
#             output = model(b_x)
#             #print(output)
#             #计算误差
#             loss = loss_fn(output, b_y)
#             #将梯度变为0
#             optimizer.zero_grad()
#             #反向传播
#             loss.backward()
#             #优化参数
#             optimizer.step()

#             #用测试集检验是否预测准确
#             if step%50 == 0:
#                 print("epoch:", epoch, "| train loss:%.4f" % loss.data)
#                 class_correct = list(0.for i in range(10))
#                 class_total = list(0.for i in range(10))
#                 for data in test_dataloader:
#                     images, labels = data
#                     outputs = model(Variable(images).to(device))
#                     labels=labels.to(device)
#                     _,predicted = torch.max(outputs.data, 1)
#                     #print(predicted)
#                     c = (predicted == labels).squeeze()
#                     for i in range(4):
#                         label = labels[i]
#                         class_correct[label] += c[i]
#                         class_total[label] += 1

#                 #for i in range(10):
#                     #print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


#                 # print("total acc= ",int(sum(class_correct))/sum(class_total))
#                 # accuracy = sum(class_correct) / sum(class_total)
#                 # accuracy=accuracy.item()
#                 # accuracies.append(accuracy)
#                 # #绘图
#                 # accuracies_np = np.array(accuracies)
#                 floss=float(loss.data)
#                 plt.plot(floss)
#                 plt.title('Model Accuracy')
#                 plt.ylabel('Accuracy')
#                 plt.xlabel('Step')
#                 plt.show() 
#     return plt.gcf()

# demo=gr.Interface(
#     train,
#     'number',
#     'plot',
#     #live=True,
# )
# demo.launch()
#train(Epoch=5)


for epoch in range(Epoch):
    #step,代表现在第几个batch_size
    #batch_x 训练集的图像
    #batch_y 训练集的标签
    for step, (batch_x, batch_y) in enumerate(train_dataloader):
        #model只接受Variable的数据，因此需要转化
        b_x=Variable(batch_x)
        b_x=b_x.to(device)
        b_y=Variable(batch_y)
        b_y=b_y.to(device)
        #将b_x输入到model得到返回值
        output = model(b_x)
        #print(output)
        #计算误差
        loss = loss_fn(output, b_y)
        #将梯度变为0
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #优化参数
        optimizer.step()

        #用测试集检验是否预测准确
        if step%50 == 0:
            print("epoch:", epoch, "| train loss:%.4f" % loss.data)
            class_correct = list(0.for i in range(10))
            class_total = list(0.for i in range(10))
            for data in test_dataloader:
                images, labels = data
                outputs = model(Variable(images).to(device))
                labels=labels.to(device)
                _,predicted = torch.max(outputs.data, 1)
                #print(predicted)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i]
                    class_total[label] += 1

            #for i in range(10):
                #print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
            print("total acc= ",int(sum(class_correct))/sum(class_total))
            accuracy = sum(class_correct) / sum(class_total)
            accuracy=accuracy.item()
            accuracies.append(accuracy)

img=cv2.imread('cat.png')
img=cv2.resize(img,(32,32))
img=np.transpose(img,(2,0,1))
img=torch.FloatTensor(img).to(device)
#img=torch.reshape(img,(1,3,32,32)),
#img=img.to(torch.float)
img = img.unsqueeze(0)
out=model(img)
_,pre = torch.max(out.data, 1)
print(pre)

accuracies_np = np.array(accuracies)
plt.plot(accuracies_np)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Step')
plt.show()            
