import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

# Function to prune Conv2D layers (structured pruning for better performance)
def prune_model(model, amount=0.1):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)  # Channel pruning
            prune.remove(module, 'weight')  # Make pruning permanent
    return model

# Load YOLO model
model = YOLO('best.pt')

# Validate the original model
results_str = model.val(data="/root/autodl-tmp/STN-YOLO/antennasets/data.yaml")
print(f"Original mAP50-95: {results_str.box.map}")

# Access the internal PyTorch model
torch_model_structured = model.model

# Apply pruning
print("Pruning model...")
pruned_torch_model_structured = prune_model(torch_model_structured, amount=0.1)
print("Model pruned.")

# Save pruned weights
save_path = 'yolov8n_structured_pruned_weights.pt'
torch.save(pruned_torch_model_structured.state_dict(), save_path)
print(f"Pruned model weights saved as '{save_path}'.")

# Reload pruned weights into a new YOLO model
pruned_model = YOLO('best.pt')  # Reload same structure
pruned_model.model.load_state_dict(torch.load(save_path), strict=False)

# Validate pruned model
results_str = pruned_model.val(data="/root/autodl-tmp/STN-YOLO/antennasets/data.yaml")
print(f"Pruned mAP50-95: {results_str.box.map}")

# Fine-tune pruned model
print("Fine-tuning pruned model...")
pruned_model.train(
    data='/root/autodl-tmp/STN-YOLO/antennasets/data.yaml',
    epochs=13,
    imgsz=640,
    batch=64,
    lr0=0.0001
)

# Save fine-tuned weights
fine_tuned_path = 'yolov8n_structured_pruned_finetuned.pt'
torch.save(pruned_model.model.state_dict(), fine_tuned_path)
print(f"Pruned and fine-tuned model saved as '{fine_tuned_path}'.")

# Validate fine-tuned model
fine_tuned_model = YOLO('best.pt')  # Load same structure again
fine_tuned_model.model.load_state_dict(torch.load(fine_tuned_path), strict=False)
results_str = fine_tuned_model.val(data="/root/autodl-tmp/STN-YOLO/antennasets/data.yaml")
print(f"Fine-tuned mAP50-95: {results_str.box.map}")
