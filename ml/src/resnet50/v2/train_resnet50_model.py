import torch
import torch.nn as nn
import torch.optim as optim
import os

from sklearn.metrics import accuracy_score

from resnet50_dataset import prepare_data
from resnet50_model import ResNet50

# Đường dẫn đến dataset
train_dir = '/content/drive/My Drive/chest_xray/train'
val_dir = '/content/drive/My Drive/chest_xray/val'
test_dir = '/content/drive/My Drive/chest_xray/test'

# Chuẩn bị DataLoader cho tập huấn luyện, kiểm tra và kiểm định
train_loader = prepare_data(train_dir, batch_size=32, is_folder=True)
val_loader = prepare_data(val_dir, batch_size=32, is_folder=True)

# Khởi tạo mô hình ResNet50
model = ResNet50(num_classes=2)  # Sử dụng mô hình đã cải tiến theo OOP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Định nghĩa loss và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
for epoch in range(10):  # Huấn luyện trong 10 epoch
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Đánh giá mô hình
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy:.4f}")

models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

os.makedirs(models_dir, exist_ok=True)

# Lưu mô hình vào thư mục 'models'
torch.save(model.state_dict(), os.path.join(models_dir, 'resnet50_model.pth'))