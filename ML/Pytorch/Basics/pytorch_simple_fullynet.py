"""
A simple walkthrough of how to code a fully connected neural network
using the PyTorch library. For demonstration we train it on the very
common MNIST dataset of handwritten digits. In this code we go through
how to create the network as well as initialize a loss function, optimizer,
check accuracy and more.

Programmed by Aladdin Persson
Translated by RehoboamX

* 2022-07-15: Initial coding
* 2022-07-15: Added more detailed comments also removed part of
              check_accuracy which would only work specifically on MNIST.
"""

# Imports导入即将被用到的包
import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset management by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!


# 在这部分我们创建一个普通的全连接神经网络，下面我们定义继承自父类nn.Module的子类NN
# 这是构建神经网络一种最常用的方式并且有很大的灵活性，有时我们还需要使用nn.Sequential让神经网络的构建变得更容易
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # 第一个全连接层fc1接收参数input_size，这里为784
        # 第二个全连接层fc2接收fc1的输出作为输入，输出num_classes维，这里为10
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        这里的 x指的是 Mnist输入图像，我们将x输入上面定义的 fc1，fc2
        我们还在其中添加了非线性激活函数 Relu（其没有参数）
        作者建议 Relu使用 nn.functional (F)里的函数来定义
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NN(784, 10)         # Mnist输入图像尺寸为28*28=784
x = torch.randn(64, 784)    # 64为设置的batchsize
print(model(x).shape)

# 设置device，有GPU时在GPU上运行，否则在CPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 神经网络超参数的具体设置和具体使用的数据集有关
# 一些超参数也可以通过实验（超参数调优）去寻找比较理想的值，如学习率lr
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# 加载训练集和测试集
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化网络
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# 损失函数（Loss）和优化器（optimizer）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练网络
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # 如果有GPU将数据放在cuda内
        data = data.to(device)
        targets = targets.to(device)

        # 将数据转换为和网络输入相符合的形状
        data = data.reshape(data.shape[0], -1)

        # 前向传播（forward）
        scores = model(data)
        loss = criterion(scores, targets)

        # 反向传播（backward）
        optimizer.zero_grad()
        loss.backward()

        # Adam优化器梯度下降
        optimizer.step()

# 查看在训练集和测试集上预测准确率来评估模型表现
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        model.train()
        return num_correct/num_samples


print(f"Accuracy on training set {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set {check_accuracy(test_loader, model)*100:.2f}")