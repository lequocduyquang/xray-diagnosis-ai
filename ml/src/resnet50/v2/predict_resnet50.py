import torch
from PIL import Image
from resnet50_model import ResNet50
from resnet50_dataset import get_transforms

# Load model
model = ResNet50(num_classes=2)
model.load_state_dict(torch.load('../models/resnet50_model.pth'))
model.eval()

# Load transforms
transform = get_transforms()

# Predict function
def predict_from_resnet50(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')  # Đảm bảo ảnh là RGB
    x = transform(image).unsqueeze(0)  # Thêm batch dimension

    # Predict
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        label = 'Normal' if pred == 0 else 'Pneumonia'
        return label, probs[0].tolist()