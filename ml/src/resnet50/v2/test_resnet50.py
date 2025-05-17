import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet50_model import ResNet50

def prepare_test_data(test_dir, batch_size=32):
    """
    Chuẩn bị dữ liệu test cho ResNet50.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def evaluate_model(model, test_loader, device):
    """
    Đánh giá mô hình trên tập test.
    """
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2%}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load mô hình ResNet50
    model = ResNet50(num_classes=2)
    model_path = '/content/drive/MyDrive/xray-diagnosis-ai/models/resnet50_model.pth'
    
    test_dir = '/content/drive/My Drive/chest_xray/test'
    # Load trọng số mô hình
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    # Chuẩn bị dữ liệu test
    test_loader = prepare_test_data(test_dir=test_dir, batch_size=32)
    # Đánh giá mô hình
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()