import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score

from resnet18_model import ResNet18
from resnet18_dataset import prepare_data

# ==== Cấu hình ====
train_dir = '/content/drive/My Drive/rare_diseases/train'
val_dir = '/content/drive/My Drive/rare_diseases/val'
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== Tải data ====
train_loader = prepare_data(train_dir, batch_size=32, is_folder=True)
val_loader = prepare_data(val_dir, batch_size=32, is_folder=True)

# ==== Khởi tạo mô hình ====
model = ResNet18(num_classes=6, use_pretrained=True, freeze_base=False)
model = model.to(device)

# ==== Loss và optimizer ====
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ==== Training loop ====
num_epochs = 20
best_val_f1 = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ==== Validation ====
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (outputs > 0.5).int().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_val_loss = val_loss / len(val_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # ==== Lưu mô hình tốt nhất ====
    if f1 > best_val_f1:
        best_val_f1 = f1
        model_path = os.path.join(models_dir, 'resnet18_multilabel.pth')
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved best model to {model_path} (Val F1: {f1:.4f})")

print(f"🏁 Training complete. Best Validation F1: {best_val_f1:.4f}")
