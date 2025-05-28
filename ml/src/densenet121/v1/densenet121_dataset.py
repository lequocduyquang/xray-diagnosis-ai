import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class XRayDataset(Dataset):
    def __init__(self, image_dir, df, label_cols, is_train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.label_cols = label_cols
        self.transform = get_transforms(is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_id'] + '.jpeg')

        # Load ảnh
        image = Image.open(img_path).convert('RGB')

        # Áp dụng transform
        image = self.transform(image)

        # Lấy label và chuyển sang tensor
        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)

        return image, labels

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # crop nhẹ tránh mất nội dung
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # xoay nhẹ
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # tránh làm lệch quá nhiều
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
