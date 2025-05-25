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

def convert_all_dicom(dicom_root, jpeg_root):
    os.makedirs(jpeg_root, exist_ok=True)

    # Duyệt tất cả file trong thư mục gốc + subfolders
    for root, _, files in os.walk(dicom_root):
        dicom_files = [f for f in files if f.lower().endswith(('.dcm', '.dicom'))]

        for fname in tqdm(dicom_files, desc=f"Converting in {root}"):
            dicom_path = os.path.join(root, fname)
            # Đặt tên file đầu ra: thay / bằng _ để tránh đụng tên
            flat_name = os.path.relpath(dicom_path, dicom_root).replace(os.sep, '_')
            jpeg_fname = os.path.splitext(flat_name)[0] + '.jpeg'
            jpeg_path = os.path.join(jpeg_root, jpeg_fname)

            dicom_to_jpeg(dicom_path, jpeg_path)

# Thư mục gốc chứa DICOM và nơi lưu JPEG
dicom_train_dir = '/content/images/train'
jpeg_train_dir = '/content/images_jpeg/train'

print("⏳ Converting all DICOM files...")
convert_all_dicom(dicom_train_dir, jpeg_train_dir)
print("✅ Conversion complete!")
