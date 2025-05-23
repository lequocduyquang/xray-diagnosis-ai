import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, num_classes=2, use_pretrained=True, freeze_base=True, dropout_rate=0.3):
        super(ResNet18, self).__init__()

        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None)

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        in_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
