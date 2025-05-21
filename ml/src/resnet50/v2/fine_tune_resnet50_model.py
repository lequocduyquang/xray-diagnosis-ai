import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision import transforms

from resnet50_model import ResNet50
from resnet50_dataset import prepare_data

# ==== Cấu hình đường dẫn và thiết bị ====
train_dir = '/content/drive/My Drive/pediatric/train'
val_dir = '/content/drive/My Drive/pediatric/val'
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== Data augmentation cho pediatric dataset ====
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

# ==== Khởi tạo DataLoader với augmentation ====
train_loader = prepare_data(train_dir, batch_size=32, is_folder=True, transform=train_transforms)
val_loader = prepare_data(val_dir, batch_size=32, is_folder=True, transform=val_transforms)

# ==== Load model đã train trên người lớn ====
model = ResNet50(num_classes=2, use_pretrained=False, freeze_base=True, dropout_rate=0.3)
model.load_state_dict(torch.load('/content/drive/My Drive/.../resnet50_model_v2.pth', map_location=device))

# ==== Chỉ fine-tune classifier (fully connected layer) ====
for param in model.base.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

# ==== Loss và optimizer với learning rate nhỏ hơn ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# ==== Huấn luyện fine-tune ====
num_epochs = 15
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
        model_path = os.path.join(models_dir, 'resnet50_model_v2_finetune_pediatric.pth')
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved best model to {model_path} (Val Acc: {val_accuracy:.4f})")

print(f"🏁 Fine-tune complete. Best Validation Accuracy: {best_val_acc:.4f}")
