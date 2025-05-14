from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def prepare_data(data_dir, batch_size=32):
    transform = get_transforms()
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # ✅ In ra mapping từ class name sang index
    print("Class to index mapping:", dataset.class_to_idx)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader