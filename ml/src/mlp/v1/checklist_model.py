import torch.nn as nn

class ChecklistMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3, num_classes=2):
        """
        Khởi tạo mô hình ChecklistMLP.
        Args:
            input_dim (int): Số chiều của dữ liệu đầu vào (số đặc trưng phi hình ảnh).
            hidden_dim (int): Số chiều của tầng ẩn.
            dropout_rate (float): Tỷ lệ dropout để giảm overfitting.
            num_classes (int): Số lượng lớp đầu ra (mặc định là 2 cho phân loại nhị phân).
        """
        super(ChecklistMLP, self).__init__()
        
        # Mô hình MLP với các tầng fully connected
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Tầng fully connected đầu tiên
            nn.BatchNorm1d(hidden_dim),        # Batch normalization
            nn.ReLU(),                         # Hàm kích hoạt ReLU
            nn.Dropout(dropout_rate),          # Dropout để giảm overfitting
            nn.Linear(hidden_dim, num_classes) # Tầng fully connected cuối cùng
        )

    def forward(self, x):
        """
        Forward pass của mô hình.
        Args:
            x (torch.Tensor): Dữ liệu đầu vào (vector đặc trưng phi hình ảnh).
        Returns:
            torch.Tensor: Đầu ra của mô hình (xác suất hoặc logits).
        """
        return self.model(x)