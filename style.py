import torch
import torch.nn as nn
from torchvision import models

class StyleResNet50(nn.Module):
  def __init__(self, hidden_dim1=1024, hidden_dim2=256):
    super(StyleResNet50, self).__init__()

    resnet = models.resnet50(pretrained=True)
    self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])

    self.fc1 = nn.Linear(2048, hidden_dim1)
    self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
    self.fc3 = nn.Linear(hidden_dim2, 27)
    self.dropout = nn.Dropout(p=0.4)
    self.flatten = nn.Flatten()
    self.ReLU = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.resnet_features(x)
    x = self.flatten(x)
    x = self.dropout(x)
    x = self.ReLU(self.fc1(x))
    x = self.dropout(x)
    x = self.ReLU(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)

    return x