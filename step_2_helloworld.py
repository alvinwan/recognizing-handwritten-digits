import torch
import torch.nn as nn
import torch.optim as optim

net = nn.Linear(1, 1)  # 1. Build a computation graph (a line!)
optimizer = optim.SGD(net.parameters(), lr=0.1)  # 2. Setup optimizers
criterion = nn.MSELoss()  # 3. Setup criterion
x, target = torch.randn((1,)), torch.tensor([0.])  # 4. Setup data

# 5. Train the model
for i in range(10):
    output = net(x)
    loss = criterion(output, target)
    print(round(loss.item(), 2))

    net.zero_grad()
    loss.backward()
    optimizer.step()