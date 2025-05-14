import torch 
import os
from dataPreprocessing import prepare_data
from modelTraining import create_model, train_model

def main():
    # Đường dẫn đến dữ liệu
    train_loader = prepare_data('/content/drive/My Drive/chest_xray/train', batch_size=32)
    val_loader = prepare_data('/content/drive/My Drive/chest_xray/val', batch_size=32)

    # Tạo và huấn luyện mô hình
    model = create_model(num_classes=2)
    trained_model = train_model(model, train_loader, val_loader, epochs=10)

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs('models', exist_ok=True)

    # Lưu mô hình
    torch.save(trained_model.state_dict(), 'models/resnet50-pneumonia.pth')
    print("Model saved to models/resnet50-pneumonia.pth")

if __name__ == "__main__":
    main()