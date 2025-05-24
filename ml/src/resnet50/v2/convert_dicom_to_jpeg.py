import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm

def dicom_to_jpeg(dicom_path, jpeg_path):
    try:
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array.astype(np.float32)

        # Basic windowing normalization
        img -= np.min(img)
        img /= np.max(img)
        img *= 255.0
        img = img.astype(np.uint8)

        # Convert grayscale to RGB (copy 3 channels)
        img = Image.fromarray(img).convert('RGB')

        # Save as JPEG with high quality
        img.save(jpeg_path, 'JPEG', quality=95)
    except Exception as e:
        print(f"⚠️ Failed to convert {dicom_path}: {e}")

def convert_folder(dicom_root, jpeg_root):
    for class_name in os.listdir(dicom_root):
        dicom_class_dir = os.path.join(dicom_root, class_name)
        if not os.path.isdir(dicom_class_dir):
            continue  # Bỏ qua file không phải thư mục

        jpeg_class_dir = os.path.join(jpeg_root, class_name)
        os.makedirs(jpeg_class_dir, exist_ok=True)

        dicom_files = [f for f in os.listdir(dicom_class_dir) if f.lower().endswith(('.dcm', '.dicom'))]

        for fname in tqdm(dicom_files, desc=f"Converting {class_name}"):
            dicom_path = os.path.join(dicom_class_dir, fname)
            jpeg_fname = os.path.splitext(fname)[0] + '.jpeg'
            jpeg_path = os.path.join(jpeg_class_dir, jpeg_fname)
            dicom_to_jpeg(dicom_path, jpeg_path)

# Thư mục gốc DICOM và thư mục lưu JPEG mới
dicom_train_dir = '/content/chest_xray_kid/train'
dicom_val_dir = '/content/chest_xray_kid/val'

jpeg_train_dir = '/content/chest_xray_kid_jpeg/train'
jpeg_val_dir = '/content/chest_xray_kid_jpeg/val'

os.makedirs(jpeg_train_dir, exist_ok=True)
os.makedirs(jpeg_val_dir, exist_ok=True)

print("⏳ Converting train folder...")
convert_folder(dicom_train_dir, jpeg_train_dir)

print("⏳ Converting val folder...")
convert_folder(dicom_val_dir, jpeg_val_dir)

print("✅ Conversion complete!")
