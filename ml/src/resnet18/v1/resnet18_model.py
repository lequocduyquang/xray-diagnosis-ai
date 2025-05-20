import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, num_classes=6, use_pretrained=True, freeze_base=True, dropout_rate=0.3):
        """
        Mô hình ResNet18 cho phân loại đa nhãn (multi-label) bệnh hiếm.

        Args:
            num_classes (int): Số lượng nhãn bệnh hiếm.
            use_pretrained (bool): Có sử dụng pretrained ImageNet không.
            freeze_base (bool): Đóng băng ResNet18 backbone hay không.
            dropout_rate (float): Tỉ lệ dropout.
        """
        super(ResNet18, self).__init__()

        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None)

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        in_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Bắt buộc dùng Sigmoid cho multi-label classification
        )

    def forward(self, x):
        return self.base_model(x)
