import os
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import warnings
from pydicom.errors import InvalidDicomError

class EfficientNetDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, check_files=True, min_samples=20):
        """
        Dataset để đọc file DICOM và nhãn multi-label từ CSV,
        lọc nhãn theo số mẫu >= min_samples.

        Args:
            csv_file (str): Đường dẫn đến file CSV chứa image_id và nhãn.
            image_dir (str): Thư mục chứa file DICOM.
            transform (callable, optional): Transform cho ảnh (e.g., torchvision.transforms).
            check_files (bool): Kiểm tra file DICOM tồn tại khi khởi tạo.
            min_samples (int): Số mẫu tối thiểu cho một nhãn được giữ lại.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        
        # Xác định cột nhãn (bỏ qua rad_ID và cột không liên quan)
        all_cols = self.df.columns
        label_cols = []
        for col in all_cols:
            if col != 'image_id' and not col.startswith('Unnamed') and col != 'rad_ID':
                try:
                    values = pd.to_numeric(self.df[col], errors='coerce')
                    if values.isin([0, 1, np.nan]).all() and not values.isna().all():
                        label_cols.append(col)
                    else:
                        print(f"⚠️ Bỏ qua cột '{col}': Chứa giá trị không nhị phân hoặc không hợp lệ.")
                except:
                    print(f"⚠️ Bỏ qua cột '{col}': Không thể chuyển thành số.")
            elif col == 'rad_ID':
                print(f"⚠️ Bỏ qua cột 'rad_ID': Là cột Radiologist ID, không liên quan đến chẩn đoán.")
        
        if not label_cols:
            raise ValueError("Không tìm thấy cột nhãn nhị phân nào trong CSV.")

        # Tính số mẫu cho mỗi nhãn
        label_counts = {}
        for label in label_cols:
            count = self.df[label].fillna(0).astype(int).sum()
            label_counts[label] = count

        # Lọc nhãn theo số mẫu min_samples
        self.label_cols = [label for label, count in label_counts.items() if count >= min_samples]
        print(f"✅ Giữ lại {len(self.label_cols)} cột nhãn (≥ {min_samples} mẫu): {self.label_cols}")

        if not self.label_cols:
            raise ValueError(f"Không còn nhãn nào sau khi lọc với min_samples={min_samples}")

        # Lọc dataframe chỉ giữ các cột image_id và nhãn còn lại
        filtered_df = self.df[['image_id'] + self.label_cols].copy()

        # Lọc những ảnh có ít nhất một nhãn dương trong nhãn còn lại
        filtered_df = filtered_df[filtered_df[self.label_cols].fillna(0).sum(axis=1) > 0].reset_index(drop=True)
        print(f"✅ Dataset sau khi lọc: còn {len(filtered_df)} ảnh có nhãn hợp lệ.")

        self.df = filtered_df
        self.num_labels = len(self.label_cols)

        # Kiểm tra file DICOM tồn tại
        if check_files:
            valid_rows = []
            missing_files = []
            for idx, row in self.df.iterrows():
                img_path = os.path.join(image_dir, f"{row['image_id']}.dicom")
                if os.path.exists(img_path):
                    valid_rows.append(row)
                else:
                    missing_files.append(row['image_id'])
            
            self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
            if missing_files:
                print(f"⚠️ Bỏ qua {len(missing_files)} file DICOM không tồn tại:")
                for f in missing_files[:5]:
                    print(f"  - {f}.dicom")
                if len(missing_files) > 5:
                    print(f"  ... và {len(missing_files) - 5} file khác.")
        
        # Lấy nhãn, xử lý giá trị NaN (nếu có)
        self.targets = self.df[self.label_cols].fillna(0).values.astype(np.float32)
        print(f"✅ Dataset loaded: {len(self.df)} ảnh hợp lệ, {self.num_labels} nhãn.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        img_path = os.path.join(self.image_dir, f"{img_id}.dicom")
        
        try:
            image = self.load_dicom_image(img_path)
        except Exception as e:
            print(f"⚠️ Lỗi đọc DICOM: {img_id}.dicom → {e}")
            image = Image.fromarray(np.zeros((224, 224), dtype=np.uint8)).convert('RGB')
            label = torch.zeros(self.num_labels, dtype=torch.float32)
            return image, label
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.FloatTensor(self.targets[idx])
        return image, label

    def load_dicom_image(self, path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
            try:
                dcm = pydicom.dcmread(path, force=True)
                if 'PixelData' not in dcm:
                    raise ValueError("File không chứa PixelData.")
                if dcm.file_meta.TransferSyntaxUID.is_compressed:
                    dcm.decompress()
                image = dcm.pixel_array.astype(np.float32)
                if len(image.shape) > 2:
                    if image.shape[-1] == 1:
                        image = image.squeeze(-1)
                    elif image.shape[-1] == 3:
                        pass
                    else:
                        raise ValueError(f"Kích thước ảnh không hợp lệ: {image.shape}")
                intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
                slope = float(getattr(dcm, 'RescaleSlope', 1.0))
                image = slope * image + intercept
                image -= image.min()
                image /= (image.max() + 1e-5)
                image = np.clip(image, 0, 1)
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                image = (image * 255).astype(np.uint8)
                return Image.fromarray(image).convert('RGB')
            except (InvalidDicomError, ValueError, Exception) as e:
                raise Exception(f"Lỗi xử lý DICOM: {str(e)}")

    def get_class_names(self):
        return self.label_cols

    def check_dicom_files(self):
        invalid_files = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.image_dir, f"{row['image_id']}.dicom")
            try:
                dcm = pydicom.dcmread(img_path, force=True)
                if 'PixelData' not in dcm:
                    invalid_files.append(img_path)
            except Exception:
                invalid_files.append(img_path)
        if invalid_files:
            print(f"⚠️ Tìm thấy {len(invalid_files)} file DICOM lỗi:")
            for f in invalid_files[:5]:
                print(f"  - {f}")
            if len(invalid_files) > 5:
                print(f"  ... và {len(invalid_files) - 5} file khác.")
        else:
            print("✅ Tất cả file DICOM hợp lệ.")
        return invalid_files
