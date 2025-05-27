import os
import pydicom
from PIL import Image
import numpy as np

# Thư mục chứa ảnh DICOM và thư mục lưu ảnh JPEG
dicom_folder = "/content/drive/MyDrive/test/train"
jpeg_folder = "/content/drive/MyDrive/test_jpeg/train"

# Tạo thư mục lưu ảnh JPEG nếu chưa tồn tại
os.makedirs(jpeg_folder, exist_ok=True)

# Duyệt qua tất cả các file trong thư mục DICOM
for filename in os.listdir(dicom_folder):
    if filename.endswith(".dicom"):
        dicom_path = os.path.join(dicom_folder, filename)
        jpeg_path = os.path.join(jpeg_folder, filename.replace(".dicom", ".jpeg"))

        # Đọc file DICOM
        dicom = pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array

        # Chuyển đổi pixel array thành ảnh và lưu dưới dạng JPEG
        image = Image.fromarray((pixel_array / np.max(pixel_array) * 255).astype(np.uint8))
        image.save(jpeg_path)

print(f"✅ Chuyển đổi hoàn tất! Ảnh JPEG được lưu tại: {jpeg_folder}")