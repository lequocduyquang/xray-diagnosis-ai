import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes=6, use_pretrained=True, freeze_base=False, dropout_rate=0.5):
        super(ResNet50, self).__init__()

        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None)

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        in_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
