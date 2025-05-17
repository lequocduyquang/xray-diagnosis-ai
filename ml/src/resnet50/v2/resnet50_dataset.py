from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

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

def get_transforms(is_train=True):
    """
    Trả về chuỗi biến đổi hình ảnh phù hợp cho huấn luyện hoặc kiểm định.
    Args:
        is_train (bool): True nếu là tập huấn luyện (có augmentation), False nếu là validation/test.
    Returns:
        torchvision.transforms.Compose: Chuỗi biến đổi hình ảnh.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

def prepare_data(data_source, batch_size=32, is_folder=True, image_col=None, label_col=None, is_train=True):
    """
    Chuẩn bị DataLoader từ thư mục hoặc từ DataFrame.

    Args:
        data_source (str hoặc pd.DataFrame): Đường dẫn thư mục chứa ảnh hoặc DataFrame chứa đường dẫn và nhãn.
        batch_size (int): Kích thước mỗi batch.
        is_folder (bool): True nếu data_source là thư mục, False nếu là DataFrame.
        image_col (str): Tên cột chứa đường dẫn ảnh (chỉ dùng nếu data_source là DataFrame).
        label_col (str): Tên cột chứa nhãn (chỉ dùng nếu data_source là DataFrame).
        is_train (bool): True nếu là tập train (sẽ có augmentation), False nếu là validation/test.

    Returns:
        DataLoader: Đối tượng DataLoader để duyệt dữ liệu theo batch.
    """
    # Lấy transform phù hợp với train hoặc val/test
    transform = get_transforms(is_train=is_train)

    # Load dataset từ thư mục
    if is_folder:
        dataset = datasets.ImageFolder(data_source, transform=transform)
        print("Class to index mapping:", dataset.class_to_idx)

    # Load dataset từ DataFrame tùy biến
    else:
        dataset = CustomImageDataset(data_source, image_col=image_col, label_col=label_col, transform=transform)

    # shuffle = True nếu là train, ngược lại không shuffle để giữ thứ tự ổn định
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

    return loader