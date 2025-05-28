import torch
import torchvision.models as models
import torch.onnx
import argparse
import os

NUM_CLASSES = 5

def load_model(model_name: str, num_classes: int):
    if model_name == "densenet121":
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")

    return model

def export_onnx(model_path, model_name, num_classes, output_path):
    print(f"📦 Loading model from {model_path}...")
    model = load_model(model_name, num_classes)
    state_dict = torch.load(model_path, map_location="cpu")

    # Xử lý nếu state_dict có tiền tố "base_model."
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("base_model."):
            new_state_dict[k.replace("base_model.", "")] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"🚀 Exporting {model_name} to {output_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print("✅ Export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to .pth model file")
    parser.add_argument("--model_name", required=True, choices=["resnet50", "densenet121"], help="Model architecture")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of output classes")
    parser.add_argument("--output_path", default="model.onnx", help="Path to save .onnx file")

    args = parser.parse_args()

    export_onnx(args.model_path, args.model_name, args.num_classes, args.output_path)
