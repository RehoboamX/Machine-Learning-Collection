"""
Shows a small example of how to load a pretrain model (VGG16) from PyTorch,
and modifies this to train on the CIFAR10 dataset. The same method generalizes
well to other datasets, but the modifications to the network may need to be changed.
Video explanation: https://youtube/U4bHxEhMGNk
Got any questions leave a comment on youtube :)
Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2022-08-06 Initial coding
"""

# imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader  # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformation we can perform on our dataset
import torchvision

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Load pretrain model & modify it
model = torchvision.models.vgg16(pretrained=True)   # 返回在ImageNet上训练好的模型
#print(model)   # print可以看到模型细节
model.avgpool = Identity()  # ImageNet上预训练的模型用在CIFAR10上不需要生成7x7大小的avgpool层（换成恒等映射）
model.classifier = nn.Linear(512, 10)   # 如果只想更改classifier中的某一层，可以通过classifier[0]等方式选择指定的层进行更改，此项也可以改成包含多层的sequential容器
model.to(device)

# Load data
train_dataset = datasets.CIFAR10(root="cifar10_data/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or Adam step
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    print(f"Loss at epoch {epoch+1} was {mean_loss:.5f}")


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


check_accuracy(train_loader, model)