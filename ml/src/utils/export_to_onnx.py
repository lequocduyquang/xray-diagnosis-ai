import torch
import torchvision.models as models
import torch.onnx
import argparse
import os

def load_model(model_name: str, num_classes: int):
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")

    return model

def export_onnx(model_path, model_name, num_classes, output_path):
    print(f"Loading model from {model_path}...")
    model = load_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Exporting {model_name} to {output_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("Export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to .pth model file")
    parser.add_argument("--model_name", required=True, choices=["resnet50", "densenet121"], help="Model architecture")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of output classes")
    parser.add_argument("--output_path", default="model.onnx", help="Path to save .onnx file")

    args = parser.parse_args()

    export_onnx(args.model_path, args.model_name, args.num_classes, args.output_path)
