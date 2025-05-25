import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class XRayDataset(Dataset):
    def __init__(self, image_dir, df, label_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.label_cols = label_cols
        self.transform = transform or get_transforms()  # fallback nếu không truyền vào

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_id'] + '.jpeg')

        # Load ảnh
        image = Image.open(img_path).convert('RGB')

        # Áp dụng transform (chuyển sang tensor)
        image = self.transform(image)

        # Lấy label và chuyển sang tensor
        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)

        return image, labels

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # quan trọng để trả về Tensor, tránh lỗi collate
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
