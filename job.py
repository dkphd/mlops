import torch
import torch.nn as nn
import wandb

config = {"epochs": 100, "w":2, "b":3}

entity = "phd-dk"
project = "mlops"

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
    

with wandb.init(
    entity=entity, config=config, project=project,
) as run:
    config = wandb.config

    x = torch.linspace(-1, 1, 100).view(-1, 1)
    y = config.w * x + config.b

    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, config.epochs):
        inputs = x
        target = y

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))
        run.log({"loss": loss, "epoch": epoch})

    print('w = {}, b = {}'.format(model.linear.weight.item(), model.linear.bias.item()))
    print(f"Should be w = {config.w}, b = {config.b}")
