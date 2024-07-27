import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

class TolubaiResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 1)

        for i in self.model.parameters():
            i.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x
