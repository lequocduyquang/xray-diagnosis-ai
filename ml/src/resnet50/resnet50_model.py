import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_layers=True, dropout_rate=0.3):
        """
        Khởi tạo mô hình ResNet50.
        Args:
            num_classes (int): Số lượng lớp đầu ra.
            pretrained (bool): Có sử dụng trọng số pre-trained hay không.
            freeze_layers (bool): Có đóng băng các tầng pre-trained hay không.
            dropout_rate (float): Tỷ lệ dropout để giảm overfitting.
        """
        super(ResNet50, self).__init__()
        
        # Load pre-trained ResNet50 model
        self.base_model = models.resnet50(pretrained=pretrained)
        
        # Freeze layers if specified
        if freeze_layers:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass của mô hình.
        Args:
            x (torch.Tensor): Dữ liệu đầu vào (ảnh).
        Returns:
            torch.Tensor: Đầu ra của mô hình.
        """
        return self.base_model(x)