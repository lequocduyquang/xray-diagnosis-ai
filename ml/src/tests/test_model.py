import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modelTraining import create_model as create_resnet50_model
from checklist_model import ChecklistMLP
import joblib

def prepare_test_data(test_dir=None, batch_size=32, model_type="resnet50", feature_cols=None):
    """
    Chuẩn bị dữ liệu test cho ResNet50 hoặc ChecklistMLP.
    """
    if model_type == "resnet50":
        # Dữ liệu test cho ResNet50
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader
    elif model_type == "checklist_mlp":
        # Dữ liệu test cho ChecklistMLP
        test_data = joblib.load(test_dir)  # Giả sử test_dir là file chứa DataFrame đã lưu
        scaler = joblib.load('models/scaler.pkl')  # Load scaler đã lưu
        test_data[feature_cols] = scaler.transform(test_data[feature_cols])
        test_features = torch.tensor(test_data[feature_cols].values, dtype=torch.float32)
        test_labels = torch.tensor(test_data['label'].values, dtype=torch.long)
        test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader
    else:
        raise ValueError("Invalid model_type. Must be 'resnet50' or 'checklist_mlp'.")

def evaluate_model(model, test_loader, device, model_type="resnet50"):
    """
    Đánh giá mô hình trên tập test.
    """
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            if model_type == "resnet50":
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
            elif model_type == "checklist_mlp":
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
            else:
                raise ValueError("Invalid model_type. Must be 'resnet50' or 'checklist_mlp'.")

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2%}")

def main():
    # Khai báo device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chọn mô hình để test
    model_type = input("Enter model type ('resnet50' or 'checklist_mlp'): ").strip().lower()

    if model_type == "resnet50":
        # Load mô hình ResNet50
        model = create_resnet50_model(num_classes=2)
        model_path = 'models/resnet50_model.pth'
        test_dir = 'chest_xray/test'
        test_loader = prepare_test_data(test_dir=test_dir, batch_size=32, model_type="resnet50")
    elif model_type == "checklist_mlp":
        # Load mô hình ChecklistMLP
        model = ChecklistMLP(input_dim=4, num_diseases=2, num_symptoms=4)  # Thay đổi input_dim và num_classes nếu cần
        model_path = 'models/checklist_mlp.pth'
        test_dir = 'data/test_data.pkl'  # Giả sử test data được lưu dưới dạng file .pkl
        feature_cols = ['fever', 'cough', 'fast_breathing', 'chest_indrawing']
        test_loader = prepare_test_data(test_dir=test_dir, batch_size=32, model_type="checklist_mlp", feature_cols=feature_cols)
    else:
        raise ValueError("Invalid model type. Must be 'resnet50' or 'checklist_mlp'.")

    # Load trọng số mô hình
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Đánh giá mô hình
    evaluate_model(model, test_loader, device, model_type=model_type)

if __name__ == "__main__":
    main()