import torch
import torch.nn as nn
from torchvision import models

class StyleResNet50_5(nn.Module):
  def __init__(self, hidden_dim1=1024, hidden_dim2=256):
    super(StyleResNet50_5, self).__init__()

    resnet = models.resnet50(pretrained=True)
    self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])

    self.fc1 = nn.Linear(2048, hidden_dim1)
    self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
    self.fc3 = nn.Linear(hidden_dim2, 13)
    self.dropout = nn.Dropout(p=0.5)
    self.flatten = nn.Flatten()
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

  def forward(self, x):
    x = self.resnet_features(x)
    x = self.flatten(x)
    x = self.dropout(x)
    x = self.leaky_relu(self.fc1(x))
    x = self.dropout(x)
    x = self.leaky_relu(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)

    return x