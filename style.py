import torch.nn as nn
from torchvision import models

class StyleResNet50_2(nn.Module):
    def __init__(self, num_classes=19, prob=0.5):
        super(StyleResNet50_2, self).__init__()
        resnet = models.resnet50(pretrained=True)
            # Freeze all parameters in ResNet
        for param in resnet.parameters():
            param.requires_grad = False
        # Get the number of input features of the last layer in ResNet
        num_features = resnet.fc.in_features
        # Remove the last fully connected layer of ResNet
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Add Dropout
        self.dropout = nn.Dropout(prob)
        # Add two fully connected layers with dropout and Leaky ReLU
        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x