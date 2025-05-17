import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes=2, use_pretrained=True, freeze_base=True, dropout_rate=0.3):
        """
        Khởi tạo mô hình ResNet50 tuỳ chỉnh.

        Args:
            num_classes (int): Số lượng lớp đầu ra (ví dụ: 2 cho phân loại nhị phân).
            use_pretrained (bool): Có sử dụng trọng số ImageNet hay không.
            freeze_base (bool): Nếu True thì đóng băng các tầng gốc ResNet để chỉ huấn luyện phần đầu ra.
            dropout_rate (float): Tỉ lệ dropout trong head classifier để giảm overfitting.
        """
        super(ResNet50, self).__init__()

        # Tải mô hình ResNet50 với trọng số ImageNet
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None)

        # Nếu muốn chỉ fine-tune phần đầu ra, đóng băng các tầng backbone
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Lấy số đầu vào của fully connected layer cuối cùng
        in_features = self.base_model.fc.in_features

        # Thay thế fc layer bằng custom head: Linear -> ReLU -> Dropout -> Linear -> Output
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Truyền dữ liệu đầu vào qua mô hình để dự đoán đầu ra.

        Args:
            x (torch.Tensor): Batch ảnh đầu vào (shape: [batch_size, 3, H, W]).

        Returns:
            torch.Tensor: Đầu ra dự đoán (shape: [batch_size, num_classes]).
        """
        return self.base_model(x)
