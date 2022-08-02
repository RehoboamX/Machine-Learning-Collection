# imports
import os
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader  # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformation we can perform on our dataset


# Simple CNN network
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    # Mnist, Input size：N x 1 x 28 x 28
    def forward(self, x):
        x = F.relu(self.conv1(x))  # N x 8 x 28 x 28
        x = self.pool(x)  # N x 8 x 14 x 14
        x = F.relu(self.conv2(x))  # N x 16 x 14 x 14
        x = self.pool(x)  # N x 16 x 7 x 7
        x = x.reshape(x.shape[0], -1)  # N x 16*7*7
        x = self.fc1(x)  # N x num_classes
        return x


def save_checkpoint(state, model_name="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    root_path = os.getcwd()
    save_path = os.path.join(root_path, "model")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model_path = os.path.join(save_path, model_name)
    torch.save(state, model_path)


def load_checkpoint(model_name):
    print("=> Loading checkpoint")
    root_path = os.getcwd()
    save_path = os.path.join(root_path, "model")
    model_path = os.path.join(save_path, model_name)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 1e-4
batch_size = 1024
num_epochs = 10
load_model = True

# Load data
train_dataset = datasets.MNIST(root="mnist_data/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="mnist_data/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint("my_checkpoint.pth")

# Train network
for epoch in range(num_epochs):
    losses = []

    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

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

