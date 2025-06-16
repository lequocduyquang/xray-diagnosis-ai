import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from densenet121_dataset import XRayDataset

# ==== MLflow (Databricks) setup ====
from dotenv import load_dotenv
import mlflow
import mlflow.pytorch

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
mlflow.set_experiment("/Users/duyquangbtx@gmail.com/densenet121_experiment")

# ==== FOCAL LOSS DEFINITION ====
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        if self.alpha is not None:
            alpha = self.alpha.view(1, -1)
            BCE_loss = alpha * BCE_loss
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==== CONFIG ====
DATA_DIR = "/content/drive/MyDrive/chest_xray_kid_multi_labels_jpeg/train"
CSV_PATH = "/content/drive/MyDrive/chest_xray_kid_multi_labels_jpeg/filtered_image_labels_train.csv"
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_CLASSES = 5
LEARNING_RATE = 1e-4

torch.manual_seed(42)
np.random.seed(42)

# ==== Load CSV và chuẩn bị labels ====
df = pd.read_csv(CSV_PATH)
label_cols = df.columns[1:].tolist()
labels = df[label_cols].values.astype(np.float32)

# ==== Tính class weights ====
label_sums = df[label_cols].sum(axis=0).values
label_counts = df.shape[0]
class_weights = label_counts / (len(label_cols) * label_sums)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# ==== Hàm tìm threshold tối ưu ====
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

# ==== Dataset và DataLoader toàn bộ data ====
dataset = XRayDataset(image_dir=DATA_DIR, df=df, label_cols=label_cols, is_train=True)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ==== Khởi tạo model ====
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model = model.to(DEVICE)

criterion = FocalLoss(alpha=class_weights, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_f1 = 0.0
best_thresholds = [0.5] * NUM_CLASSES

# ==== MLflow tracking ====
with mlflow.start_run():
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("num_epochs", NUM_EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("gamma", 2)
    mlflow.log_param("model", "densenet121")
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss", "focal_loss")

    # Train trên toàn bộ data
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(data_loader)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f}")
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

    # Lưu model final local
    save_path = os.path.join(MODEL_DIR, "densenet121_final_full_data.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved final model trained on full data at {save_path}")

    # Log model lên MLflow Databricks Model Registry (Unity Catalog)
    mlflow.pytorch.log_model(
        model,
        name="lakehouse_local.default.densenet121_classifier"
    )