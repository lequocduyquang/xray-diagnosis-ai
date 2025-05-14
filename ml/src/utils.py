import torch

def export_to_onnx(model, output_path):
    """
    Xuất mô hình PyTorch sang định dạng ONNX.
    
    Args:
        model (torch.nn.Module): Mô hình PyTorch đã được huấn luyện.
        output_path (str): Đường dẫn để lưu file ONNX.
    """
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