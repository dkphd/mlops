import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
    

x = torch.linspace(-1, 1, 100).view(-1, 1)
y = 2 * x + 3

model = LinearModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    inputs = x
    target = y

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

print('w = {}, b = {}'.format(model.linear.weight.item(), model.linear.bias.item()))
print(f"Should be w = 2, b = 3")
torch.save(model.state_dict(), 'model.pth')
