import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
from torchvision import transforms
from efficientnet_dataset import EfficientNetDataset
from efficientnet_model import EfficientNetClassifier
from focal_loss import FocalLoss

# ==== CONFIG ====
csv_path = "/content/drive/MyDrive/chest_xray_children/image_labels_train.csv"
image_dir = "/content/drive/MyDrive/chest_xray_children/train"
save_model_path = "/content/drive/MyDrive/chest_xray_children/efficientnet_xray_best.pth"
batch_size = 16  # Giảm để tối ưu bộ nhớ trên Colab
lr = 1e-4
num_epochs = 20
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tune_epoch = 10  # Bắt đầu fine-tune sau epoch 10

# ==== TRANSFORM ====
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==== DATA ====
dataset = EfficientNetDataset(csv_file=csv_path, image_dir=image_dir, transform=None)
dataset.check_dicom_files()  # Kiểm tra file DICOM lỗi
num_classes = len(dataset.get_class_names())

# Tính class weights
num_samples = len(dataset)
label_counts = dataset.targets.sum(axis=0)
class_weights = num_samples / (2.0 * label_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)

# In phân bố nhãn
def print_label_distribution(dataset):
    label_sums = dataset.targets.sum(axis=0)
    print("Phân bố nhãn:")
    for name, count in zip(dataset.get_class_names(), label_sums):
        print(f"  - {name}: {int(count)} mẫu ({count/len(dataset.targets)*100:.2f}%)")

print_label_distribution(dataset)

# Chia train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_ds.dataset.transform = train_transform
val_ds.dataset.transform = val_transform

# DataLoader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

# ==== MODEL ====
model = EfficientNetClassifier(
    num_classes=num_classes,
    model_name='efficientnet-b0',
    freeze_backbone=True,  # Đóng băng backbone ban đầu
    dropout_rate=0.3
).to(device)

criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean', device=device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
scaler = torch.cuda.amp.GradScaler()

# ==== TRAINING ====
best_val_f1 = 0
epochs_no_improve = 0
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

for epoch in range(num_epochs):
    # Fine-tune theo từng giai đoạn
    if epoch == 10:
        print("🔄 Epoch 10: Mở 20% backbone")
        model.unfreeze_backbone(unfreeze_ratio=0.2)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr/10)
    elif epoch == 13:
        print("🔄 Epoch 13: Mở 50% backbone")
        model.unfreeze_backbone(unfreeze_ratio=0.5)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr/20)
    elif epoch == 16:
        print("🔄 Epoch 16: Mở 80% backbone")
        model.unfreeze_backbone(unfreeze_ratio=0.8)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr/50)
    
    # Train
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.3
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
    
    # Tính F1-score
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    val_f1 = f1_score(all_labels, all_preds, average="micro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Cập nhật history
    history['train_loss'].append(train_loss / len(train_loader))
    history['val_loss'].append(val_loss / len(val_loader))
    history['val_f1'].append(val_f1)
    
    # Scheduler step
    scheduler.step(val_f1)
    
    # In kết quả
    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Val F1 (micro): {val_f1:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    print("F1 per class:", {name: f"{f1:.4f}" for name, f1 in zip(dataset.get_class_names(), f1_per_class)})
    
    # Lưu model tốt nhất
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), save_model_path)
        print(f"✅ Saved best model to {save_model_path} (Val F1: {val_f1:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⏹ Early stopping.")
            break

print("🏁 Training complete. Best Val F1:", best_val_f1)