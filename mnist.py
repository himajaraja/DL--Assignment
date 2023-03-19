import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the neural network
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set up the dataset and dataloader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Initialize the model, loss function, and optimizer
model = FNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model and plot the loss and accuracy curves
num_epochs = 10
train_loss = []
test_loss = []
train_acc = []
test_acc = []
for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        running_acc += torch.sum(preds == labels.data).float() / len(labels)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    with torch.no_grad():
        running_loss = 0.0
        running_acc = 0.0
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            running_acc += torch.sum(preds == labels.data).float() / len(labels)
        epoch_loss = running_loss / len(test_loader)
        epoch_acc = running_acc / len(test_loader)
        test_loss.append(epoch_loss)
        test_acc.append(epoch_acc)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch + 1, num_epochs, train_loss[-1], train_acc[-1], test_loss[-1], test_acc[-1]))

# Plot the loss and accuracy curves
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='train loss')
plt.plot(test_loss, label='test loss')
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(train_acc, label='train accuracy')
plt.plot(test_acc, label='test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
