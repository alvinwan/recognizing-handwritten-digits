import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# 1. Build a computation graph
net = models.ResNet18()

optimizer = optim.Adadelta(net.parameters(), lr=1.)  # 2. Setup optimizer
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
criterion = nn.NLLLoss()  # 3. Setup criterion

# 4. Setup data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(
    'data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512)
val_dataset = datasets.MNIST(
    'data', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512)

# 5. Train the model
for epoch in range(10):
    print(f'==> Epoch {epoch}')

    for (inputs, target) in train_loader:
        output = net(inputs)
        loss = criterion(output, target)
        print(round(loss.item(), 2))

        net.zero_grad()
        loss.backward()
        optimizer.step()

correct = 0.
net.eval()
for (inputs, target) in val_loader:
    output = net(inputs)
    _, pred = output.max(1)
    correct += (pred == target).sum()
accuracy = correct / len(val_dataset) * 100.
print(f'{accuracy:.2f}% correct')