import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from resnet18_model import ResNet18
from resnet18_dataset import prepare_data

train_dir = '/content/sample_data/chest_xray/train'
val_dir = '/content/sample_data/chest_xray/val'
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ==== Khởi tạo DataLoader ====
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

train_loader = prepare_data(train_dir, batch_size=32, transform=transform)
val_loader = prepare_data(val_dir, batch_size=32, transform=transform)

# ==== Khởi tạo mô hình ====
model = ResNet18(num_classes=2, use_pretrained=True, freeze_base=False, dropout_rate=0.3)
model = model.to(device)

# ==== Loss + Optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ==== Huấn luyện ====
num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ==== Validation ====
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_accuracy = accuracy_score(all_labels, all_preds)
    avg_val_loss = total_val_loss / len(val_loader)

    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_accuracy:.4f}")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        model_path = os.path.join(models_dir, 'resnet18_best.pth')
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved best model to {model_path} (Val Acc: {val_accuracy:.4f})")

print(f"🏁 Training complete. Best Validation Accuracy: {best_val_acc:.4f}")
