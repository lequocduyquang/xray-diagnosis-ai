import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from efficientnet_pytorch import EfficientNet
from efficientnet_dataset import EfficientNetDataset
from tqdm import tqdm

# ==== CONFIG ====
csv_file = "/content/drive/My Drive/chest_xray_kid_multi_labels_jpeg/filtered_image_labels_train.csv"
image_dir = "/content/drive/My Drive/chest_xray_kid_multi_labels_jpeg/train"
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

batch_size = 32
num_epochs = 20
num_workers = 2
n_splits = 3  # Giảm fold từ 5 xuống 3
learning_rate = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== TRANSFORMS ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==== Focal Loss cho multi-label ====
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        probas = torch.sigmoid(inputs)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==== Hàm lấy nhãn đa nhãn dạng 1D array để StratifiedKFold nhóm theo multi-label ====
def multilabel_stratify_labels(dataset):
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels.numpy())
    all_labels = np.array(all_labels)
    # Chuyển mỗi nhãn đa nhãn thành chuỗi bit để tạo stratify key
    # Ví dụ: [0,1,0,1] -> '0101'
    keys = [''.join(map(str, map(int, row))) for row in all_labels]
    return keys

def print_label_distribution(dataset, dataset_name):
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels.numpy())
    all_labels = np.array(all_labels)
    label_sums = np.sum(all_labels, axis=0)
    print(f"\nLabel distribution in {dataset_name}:")
    for idx, count in enumerate(label_sums):
        print(f"  Class {idx}: {int(count)} samples")

def prepare_fold_dataloaders(dataset, train_idx, val_idx):
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    print_label_distribution(train_dataset, "Train Set")
    print_label_distribution(val_dataset, "Validation Set")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def build_model(num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    return model.to(device)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().astype(int))

    val_loss /= len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return val_loss, val_f1

def save_model(model, epoch, val_f1, path):
    torch.save(model.state_dict(), path)
    print(f"✅ Saved best model at epoch {epoch+1} | Val F1: {val_f1:.4f} | Path: {path}")

def filter_rare_samples(dataset, num_classes, min_samples=10):
    """
    Lọc bỏ những mẫu có chứa nhãn nào có tổng số mẫu < min_samples.
    Trả về danh sách index mẫu được giữ lại.
    """
    # Tính tổng số mẫu mỗi class
    label_counts = np.zeros(num_classes, dtype=int)
    for _, labels in dataset:
        label_counts += labels.numpy().astype(int)
    
    # Tìm những class quá hiếm
    rare_classes = [i for i, count in enumerate(label_counts) if count < min_samples]
    print(f"Rare classes (count<{min_samples}): {rare_classes}")

    # Lọc index mẫu không chứa nhãn thuộc rare_classes
    valid_indices = []
    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        labels_np = labels.numpy().astype(int)
        if not any(labels_np[i] == 1 for i in rare_classes):
            valid_indices.append(idx)
    print(f"Samples before filtering: {len(dataset)}, after filtering: {len(valid_indices)}")
    return valid_indices

def main():
    dataset = EfficientNetDataset(csv_file, image_dir, transform=transform)
    num_classes = dataset[0][1].shape[0]  # Lấy số nhãn trực tiếp từ shape nhãn đầu tiên
    print(f"Number of classes: {num_classes}")

    # Lọc mẫu quá hiếm
    valid_indices = filter_rare_samples(dataset, num_classes=num_classes, min_samples=10)
    
    # Tạo dataset con chứa mẫu hợp lệ
    filtered_dataset = Subset(dataset, valid_indices)

    # Lấy stratify keys trên filtered dataset
    stratify_keys = multilabel_stratify_labels(filtered_dataset)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_f1_overall = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(filtered_dataset)), stratify_keys)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        train_loader, val_loader = prepare_fold_dataloaders(filtered_dataset, train_idx, val_idx)
        model = build_model(num_classes)

        criterion = FocalLoss(alpha=0.25, gamma=2)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_f1 = 0.0
        for epoch in range(num_epochs):
            print(f"\n🌀 Epoch {epoch+1}/{num_epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_f1 = validate(model, val_loader, criterion)

            print(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1 (weighted): {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                save_model(model, epoch, val_f1, os.path.join(models_dir, f"efficientnet_b0_fold{fold+1}.pth"))

        if best_f1 > best_f1_overall:
            best_f1_overall = best_f1

    print(f"\n🎉 Training completed. Best overall weighted F1: {best_f1_overall:.4f}")

if __name__ == "__main__":
    main()
