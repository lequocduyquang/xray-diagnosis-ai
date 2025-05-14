import torch
from torchvision import transforms
from PIL import Image
from modelTraining import create_model  # Import hàm tạo mô hình

# 1. Load mô hình đã huấn luyện
model_path = "models/resnet50-pneumonia.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tạo mô hình và load trọng số
model = create_model(num_classes=2)  # Hàm create_model từ file modelTraining.py
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 2. Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize ảnh về 224x224
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Chuẩn hóa ảnh
                         std=[0.229, 0.224, 0.225])
])

# Đường dẫn ảnh cần test
image_path = "images/test/NORMAL2-IM-1427-0001.jpeg"  # Thay bằng đường dẫn ảnh của bạn
img = Image.open(image_path).convert("RGB")  # Đảm bảo ảnh là RGB
img_tensor = transform(img).unsqueeze(0).to(device)  # Thêm batch dimension [1, 3, 224, 224]

print("Tensor sau transform (Python):", img_tensor[0, :, :5, :5])  # In 5x5 pixel đầu

# 3. Dự đoán
with torch.no_grad():
    output = model(img_tensor)  # Chạy inference
    probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Tính xác suất
    predicted_class = torch.argmax(probabilities).item()  # Lấy lớp dự đoán

# 4. Mapping nhãn
class_labels = ["NORMAL", "PNEUMONIA"]  # Nhãn tương ứng với các lớp
print(f"Prediction: {class_labels[predicted_class]}")
print(f"Probabilities: {probabilities.cpu().numpy()}")