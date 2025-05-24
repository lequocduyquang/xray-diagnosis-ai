import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from resnet18_model import ResNet18
from resnet18_dataset import ResNet18Dataset

# ==== Đặt seed để reproducible ====
def set_seed(seed=42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ==== Cấu hình đường dẫn và thiết bị ====
train_dir = '/content/chest_xray_kid/train'
val_dir = '/content/chest_xray_kid/val'
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ==== Transform ====
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_loader = ResNet18Dataset(train_dir, batch_size=32, transform=transform)
val_loader = ResNet18Dataset(val_dir, batch_size=32, transform=transform)

# ==== Mô hình ====
model = ResNet18(num_classes=2, use_pretrained=True, freeze_base=False, dropout_rate=0.3)
model = model.to(device)

# ==== Loss + Optimizer + LR Scheduler ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # giảm lr mỗi 5 epochs

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# ==== Training loop ====
num_epochs = 20
best_val_acc = 0.0
patience = 5
early_stop_counter = 0

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    total_train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Optional: Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        scaler.step(optimizer)
        scaler.update()

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
    scheduler.step()

    elapsed = time.time() - start_time
    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_accuracy:.4f} | "
          f"Time: {elapsed:.1f}s")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        early_stop_counter = 0
        model_path = os.path.join(models_dir, f'resnet18_best_{val_accuracy:.4f}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved best model to {model_path}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

print(f"🏁 Training complete. Best Validation Accuracy: {best_val_acc:.4f}")
