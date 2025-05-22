import os
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
from PIL import Image

# Tự viết lại Dataset đọc từ folder chứa file DICOM
class FineTuneResNet50Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}

        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.endswith('.dicom') or fname.endswith('.dcm'):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path, label = self.samples[idx]
        dcm = pydicom.dcmread(dcm_path)
        img_array = dcm.pixel_array.astype(np.float32)

        # Normalize về 0–255
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255.0
        img = Image.fromarray(img_array).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


def prepare_data(data_dir, batch_size=32, is_folder=True, transform=None):
    dataset = FineTuneResNet50Dataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
