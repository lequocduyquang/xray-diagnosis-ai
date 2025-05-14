from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class CustomImageDataset(Dataset):
    def __init__(self, df, image_col, label_col, transform=None):
        """
        Dataset tùy chỉnh để hỗ trợ DataFrame.
        Args:
            df (pd.DataFrame): DataFrame chứa đường dẫn ảnh và nhãn.
            image_col (str): Tên cột chứa đường dẫn ảnh.
            label_col (str): Tên cột chứa nhãn.
            transform (torchvision.transforms.Compose): Các phép biến đổi áp dụng lên ảnh.
        """
        self.df = df
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform if transform else get_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx][self.image_col]
        label = self.df.iloc[idx][self.label_col]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_data(data_source, batch_size=32, is_folder=True, image_col=None, label_col=None):
    """
    Chuẩn bị dữ liệu từ thư mục hoặc DataFrame.
    Args:
        data_source (str hoặc pd.DataFrame): Đường dẫn thư mục hoặc DataFrame chứa dữ liệu.
        batch_size (int): Kích thước batch.
        is_folder (bool): True nếu data_source là thư mục, False nếu là DataFrame.
        image_col (str): Tên cột chứa đường dẫn ảnh (nếu data_source là DataFrame).
        label_col (str): Tên cột chứa nhãn (nếu data_source là DataFrame).
    Returns:
        DataLoader: DataLoader chứa dữ liệu đã được chuẩn bị.
    """
    transform = get_transforms()

    if is_folder:
        dataset = datasets.ImageFolder(data_source, transform=transform)
        print("Class to index mapping:", dataset.class_to_idx)
    else:
        dataset = CustomImageDataset(data_source, image_col=image_col, label_col=label_col, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader