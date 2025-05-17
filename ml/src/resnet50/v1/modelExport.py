import torch
from modelTraining import create_model

def export_to_onnx(model, output_path):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input size
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    # Load the trained model
    model = create_model(num_classes=2)
    model.load_state_dict(torch.load('models/resnet50-pneumonia.pth'))
    
    # Export the model to ONNX format
    export_to_onnx(model, 'models/resnet50-pneumonia.onnx')