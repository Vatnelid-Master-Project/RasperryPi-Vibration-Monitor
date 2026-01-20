import torch
import onnxruntime
# Load your model
checkpoint = torch.load('autoencoder.pkl', map_location='cpu', weights_only=False)
model = checkpoint.model
model.eval()

# Create dummy input (adjust shape to your model)
dummy_input = torch.randn(1, 3, 112, 112)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "autoencoder.onnx",
    export_params=True,
    input_names=['input'],
    output_names=['output']
)
print("Model exported to autoencoder.onnx")