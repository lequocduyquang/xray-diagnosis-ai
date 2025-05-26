import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from resnet50_dataset import XRayDataset
from resnet50_model import ResNet50

# ==== FOCAL LOSS DEFINITION ====
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor kích thước (num_classes,) hoặc None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = sigmoid(logits) nếu target đúng

        # Apply class weights nếu có
        if self.alpha is not None:
            alpha = self.alpha.view(1, -1)  # reshape để broadcast đúng với batch
            BCE_loss = alpha * BCE_loss

        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==== Cấu hình ====
DATA_DIR = "/content/drive/MyDrive/chest_xray_kid_multi_labels_jpeg/train"
CSV_PATH = "/content/drive/MyDrive/chest_xray_kid_multi_labels_jpeg/filtered_image_labels_train.csv"
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_CLASSES = 5
LEARNING_RATE = 1e-4
N_SPLITS = 3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==== Load CSV và chuẩn bị labels ====
df = pd.read_csv(CSV_PATH)
label_cols = df.columns[1:].tolist()
labels = df[label_cols].values.astype(np.float32)

# ==== Tính class weights ==== 
label_sums = df[label_cols].sum(axis=0).values  # Tổng số ảnh dương cho mỗi class
label_counts = df.shape[0]
class_weights = label_counts / (len(label_cols) * label_sums)  # Công thức: N / (C * n_i)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)


# ==== Hàm tìm threshold tối ưu cho từng class ====
def find_best_thresholds(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_thresholds = []
    for i in range(y_true.shape[1]):
        f1_scores = []
        for t in thresholds:
            preds = (y_probs[:, i] > t).astype(int)
            f1_scores.append(f1_score(y_true[:, i], preds))
        best_t = thresholds[np.argmax(f1_scores)]
        best_thresholds.append(best_t)
    return best_thresholds

# ==== Khởi tạo MultilabelStratifiedKFold ====
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# ==== Training loop với K-Fold ====
for fold, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(labels)), labels), 1):
    print(f"\n===== Fold {fold}/{N_SPLITS} =====")

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = XRayDataset(image_dir=DATA_DIR, df=df_train, label_cols=label_cols, is_train=True)
    val_dataset = XRayDataset(image_dir=DATA_DIR, df=df_val, label_cols=label_cols, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = ResNet50(num_classes=NUM_CLASSES, use_pretrained=True, freeze_base=False, dropout_rate=0.3)
    model = model.to(DEVICE)

    criterion = FocalLoss(alpha=class_weights, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0.0
    best_thresholds = [0.5] * NUM_CLASSES  # Khởi tạo threshold mặc định

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

                preds = torch.sigmoid(outputs)  # sigmoid trước khi threshold
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        avg_val_loss = val_loss / len(val_loader)
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Tìm threshold tối ưu trên val set
        best_thresholds = find_best_thresholds(all_targets, all_preds)

        # Áp threshold riêng cho từng class để ra nhãn dự đoán cuối cùng
        preds_binary = np.zeros_like(all_preds, dtype=int)
        for i in range(NUM_CLASSES):
            preds_binary[:, i] = (all_preds[:, i] > best_thresholds[i]).astype(int)

        val_f1 = f1_score(all_targets, preds_binary, average='weighted')

        print(f"[Fold {fold}][Epoch {epoch}/{NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
        print(f" Thresholds: {np.round(best_thresholds, 3)}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(MODEL_DIR, f"resnet50_fold{fold}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model fold {fold} at epoch {epoch} with Val F1: {val_f1:.4f}")

print("🏁 Training complete.")
