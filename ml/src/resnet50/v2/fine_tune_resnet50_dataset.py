import os
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
from PIL import Image

class FineTuneResNet50Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}

        for class_name in self.class_to_idx:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ Warning: {class_dir} not found, skipping.")
                continue

            for fname in os.listdir(class_dir):
                if fname.endswith('.dicom') or fname.endswith('.dcm'):
                    fpath = os.path.join(class_dir, fname)
                    self.samples.append((fpath, self.class_to_idx[class_name]))

        print(f"📦 Loaded {len(self.samples)} DICOM samples from: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path, label = self.samples[idx]

        if not os.path.exists(dcm_path):
            raise FileNotFoundError(f"❌ DICOM file not found: {dcm_path}")

        dcm = pydicom.dcmread(dcm_path)
        img_array = dcm.pixel_array.astype(np.float32)

        # Normalize to [0, 255]
        img_array -= img_array.min()
        img_array /= img_array.max()
        img_array *= 255.0

        img = Image.fromarray(img_array).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


def prepare_data(data_dir, batch_size=32, is_folder=True, transform=None):
    dataset = FineTuneResNet50Dataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
