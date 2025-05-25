import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from resnet50_dataset import XRayDataset
from resnet50_model import ResNet50

# ==== Cấu hình ====
DATA_DIR = "/content/drive/My Drive/chest_xray_kid_multi_labels_jpeg/train"
CSV_PATH = "/content/drive/My Drive/chest_xray_kid_multi_labels_jpeg/filtered_image_labels_train.csv"
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_CLASSES = 6
LEARNING_RATE = 1e-4
N_SPLITS = 3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==== Load CSV và chuẩn bị labels ====
df = pd.read_csv(CSV_PATH)
label_cols = df.columns[1:].tolist()

# Kiểm tra và chuẩn bị label matrix
labels = df[label_cols].values.astype(np.float32)

# ==== Dataset full (chưa chia train/val) ====
full_dataset = XRayDataset(image_dir=DATA_DIR, df=df, label_cols=label_cols)

# ==== Khởi tạo MultilabelStratifiedKFold ====
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# ==== Training loop với K-Fold ====
for fold, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(labels)), labels), 1):
    print(f"\n===== Fold {fold}/{N_SPLITS} =====")

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Khởi tạo model mới cho mỗi fold
    model = ResNet50(num_classes=NUM_CLASSES, use_pretrained=True, freeze_base=False, dropout_rate=0.3)
    model = model.to(DEVICE)

    criterion = nn.BCELoss()  # Binary Cross-Entropy cho multi-label với sigmoid output
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        avg_val_loss = val_loss / len(val_loader)
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Dùng threshold 0.5 để đánh giá F1 score đa nhãn
        preds_binary = (all_preds > 0.5).astype(int)
        val_f1 = f1_score(all_targets, preds_binary, average='weighted')

        print(f"[Fold {fold}][Epoch {epoch}/{NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Lưu model tốt nhất
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(MODEL_DIR, f"resnet50_fold{fold}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model fold {fold} at epoch {epoch} with Val F1: {val_f1:.4f}")

print("🏁 Training complete.")
