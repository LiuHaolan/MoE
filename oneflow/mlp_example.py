
import torch
from torch import nn
from torch.optim import Adam

input_size = 1000
hidden_size = 64
batch_size = 5
num_classes = 20

def dummy_data(batch_size, input_size, num_classes):
    # dummy input
    x = torch.rand(batch_size, input_size)

    # dummy target
    y = torch.randint(low=0,high=num_classes, size=(batch_size, 1)).squeeze(1)
    return x,y

def train(x,y, model, loss_fn, optim):
    y_hat = model(x.float())
    # calculate prediction loss
    loss = loss_fn(y_hat, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    return model

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.log_soft(out)
        return out

model = MLP(input_size, num_classes, hidden_size)
x, y = dummy_data(batch_size, input_size, num_classes)
