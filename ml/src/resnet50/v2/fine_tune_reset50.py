import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

from resnet50_dataset import prepare_data  # Trả về Dataset, không phải DataLoader

# ==== CONFIG ====
train_dir_child = '/content/drive/My Drive/chest_xray_children/v1'
checkpoint_path = '/path/to/models/resnet50_model_v2.pth'

models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== DATA AUGMENTATION ====
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==== LOAD DỮ LIỆU ====
full_dataset = prepare_data(train_dir_child, batch_size=None, is_folder=True, is_train=True, transform=transform_train)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Cập nhật transform cho validation
val_dataset.dataset.transform = transform_val

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# ==== LOAD MÔ HÌNH ====
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)

# Load checkpoint từ training trước đó (người lớn)
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)

# ==== FREEZE BASE ====
for name, param in model.named_parameters():
    if "fc" not in name:  # freeze base
        param.requires_grad = False

# ==== LOSS + OPTIMIZER ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# ==== TRAIN LOOP ====
num_epochs = 30
patience = 5
best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # === Validation ===
    model.eval()
    total_val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    avg_val_loss = total_val_loss / len(val_loader)
    scheduler.step(val_acc)

    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")

    # Early stopping & checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        best_model_path = os.path.join(models_dir, 'resnet50_finetuned_best.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved best model to {best_model_path} (Val Acc: {val_acc:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⏹ Early stopping triggered.")
            break

print(f"🏁 Fine-tuning complete. Best Validation Accuracy: {best_val_acc:.4f}")
