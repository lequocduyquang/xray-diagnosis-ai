import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class EfficientNetDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Đường dẫn file CSV chứa image_id và nhãn multi-label (float 0.0 hoặc 1.0).
            image_dir (str): Thư mục chứa ảnh, ảnh có tên theo image_id + .png (hoặc .jpg).
            transform (callable, optional): Transform áp dụng cho ảnh.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        
        # Lấy tên các nhãn (cột từ thứ 2 trở đi)
        self.class_names = list(self.data.columns[1:])
        
        # Lấy nhãn dạng numpy array, float32
        self.targets = self.data[self.class_names].values.astype('float32')
        
        # Lấy danh sách image_ids
        self.image_ids = self.data['image_id'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Lấy image id và đường dẫn ảnh
        image_id = self.image_ids[idx]
        # Giả sử ảnh là .png, bạn chỉnh sửa nếu ảnh dạng khác
        img_path = os.path.join(self.image_dir, f"{image_id}.jpeg")
        
        # Mở ảnh
        image = Image.open(img_path).convert('RGB')
        
        # Áp transform nếu có
        if self.transform:
            image = self.transform(image)
        
        # Lấy nhãn multi-label (tensor float)
        labels = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return image, labels

    def get_class_names(self):
        return self.class_names

    def check_dicom_files(self):
        """
        Nếu dataset có DICOM files, bạn tự triển khai kiểm tra lỗi.
        Ở đây giả sử ảnh là PNG nên không cần kiểm tra.
        """
        pass
