from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, df, image_col, label_col, transform=None):
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

def remove_ds_store_files(folder_path):
    count = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename == '.DS_Store':
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                count += 1
    print(f"🧹 Removed {count} .DS_Store files.")

def prepare_data(data_source, batch_size=32, is_folder=True, image_col=None, label_col=None, is_train=True):
    transform = get_transforms(is_train=is_train)

    if is_folder:
        remove_ds_store_files(data_source)

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

        def is_valid_file(path):
            return path.lower().endswith(valid_extensions)

        dataset = datasets.ImageFolder(data_source, transform=transform, is_valid_file=is_valid_file)
        print("Class to index mapping:", dataset.class_to_idx)

    else:
        dataset = CustomImageDataset(data_source, image_col=image_col, label_col=label_col, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

    return loader