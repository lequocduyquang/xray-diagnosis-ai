import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from resnet50_model import ResNet50
from fine_tune_resnet50_dataset import FineTuneResNet50Dataset

# ==== Cấu hình đường dẫn và thiết bị ====
train_dir = '/content/chest_xray_kid/train'
val_dir = '/content/chest_xray_kid/val'
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ==== Data augmentation ====
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Load dataset ====
train_dataset = FineTuneResNet50Dataset(train_dir, transform=train_transforms)
val_dataset = FineTuneResNet50Dataset(val_dir, transform=val_transforms)

# ==== Dataloader ====
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ==== Load model pretrained & fine-tune ====
model = ResNet50(num_classes=2, use_pretrained=True, freeze_base=True, dropout_rate=0.3)
model.load_state_dict(torch.load('/content/drive/My Drive/xray-diagnosis-ai/resnet50/v2/models/resnet50_model_v2.pth', map_location=device))

# Freeze toàn bộ base, chỉ fine-tune fc lúc đầu
for param in model.base_model.parameters():
    param.requires_grad = False
for param in model.base_model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# ==== Tính class weights từ toàn bộ tập train ====
labels = [label for _, label in train_dataset]
class_counts = Counter(labels)
total = sum(class_counts.values())
weights = [total / class_counts[i] for i in sorted(class_counts.keys())]

class_weights = torch.FloatTensor(weights).to(device)
print("Class weights:", class_weights.tolist())

# ==== Loss, optimizer và scheduler ====
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# ==== Huấn luyện ====
num_epochs = 15
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    # 👉 Mở khóa layer4 sau epoch 5
    if epoch == 5:
        for param in model.base_model.layer4.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
        print("🔓 Unfroze layer4 for fine-tuning.")

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
    scheduler.step(avg_val_loss)

    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_accuracy:.4f}")

    # Lưu best model
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        model_path = os.path.join(models_dir, 'resnet50_model_v2_finetune.pth')
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved best model to {model_path} (Val Acc: {val_accuracy:.4f})")

print(f"\n🏁 Fine-tune complete. Best Validation Accuracy: {best_val_acc:.4f}")
