import os
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
from PIL import Image


class ResNet18Dataset(Dataset):
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
                if fname.lower().endswith(('.dicom', '.dcm')):
                    fpath = os.path.join(class_dir, fname)
                    self.samples.append((fpath, self.class_to_idx[class_name]))

        print(f"📦 Loaded {len(self.samples)} DICOM samples from: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path, label = self.samples[idx]

        if not os.path.exists(dcm_path):
            raise FileNotFoundError(f"❌ DICOM file not found: {dcm_path}")

        try:
            dcm = pydicom.dcmread(dcm_path)
            img_array = dcm.pixel_array.astype(np.float32)

            # Normalize to [0, 255] & convert to uint8
            img_array -= img_array.min()
            img_array /= (img_array.max() + 1e-5)
            img_array *= 255.0
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array).convert('RGB')

        except Exception as e:
            print(f"❌ Error reading DICOM: {dcm_path}, {e}")
            img = Image.new('RGB', (224, 224))  # fallback image

        if self.transform:
            img = self.transform(img)

        return img, label
