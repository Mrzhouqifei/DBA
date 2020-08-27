import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):

  def __init__(self):
    super(MnistModel, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    # feature map size is 14*14 by pooling
    # padding=2 for same padding
    self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    # feature map size is 7*7 by pooling
    self.fc1 = nn.Linear(64 * 7 * 7, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x, dba=False):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 64 * 7 * 7)  # reshape Variable
    h = self.fc1(x)
    x = F.relu(h)
    x = self.fc2(x)
    if dba:
      return x, h
    else:
      return x
    # return F.log_softmax(x, dim=-1)


class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(1024 * 4, 256)
    self.fc2 = nn.Linear(256, 2)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
    # return F.log_softmax(x, dim=-1)