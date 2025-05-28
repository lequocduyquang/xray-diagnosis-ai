import torch

def predict_image(img_tensor, model_resnet18, model_efficientnet, rare_label_indices, threshold=0.5):
    """
    Sequential prediction logic:
    1. Dùng ResNet18 predict trước cho bệnh hiếm
    2. Nếu có bệnh hiếm → return
    3. Ngược lại → chạy EfficientNet để predict bệnh phổ biến
    """

    # Ensure model is in eval mode
    model_resnet18.eval()
    model_efficientnet.eval()

    with torch.no_grad():
        # Bước 1: predict bằng ResNet18
        output_rare = model_resnet18(img_tensor.unsqueeze(0))  # add batch dim
        pred_rare = torch.sigmoid(output_rare).squeeze(0)

        # Kiểm tra nếu có nhãn bệnh hiếm vượt ngưỡng
        has_rare_disease = (pred_rare[rare_label_indices] > threshold).any()

        if has_rare_disease:
            return {
                "source": "resnet18",
                "prediction": pred_rare,
                "rare_detected": True
            }

        # Bước 2: nếu không có → chạy EfficientNet
        output_common = model_efficientnet(img_tensor.unsqueeze(0))
        pred_common = torch.sigmoid(output_common).squeeze(0)

        return {
            "source": "efficientnet",
            "prediction": pred_common,
            "rare_detected": False
        }
