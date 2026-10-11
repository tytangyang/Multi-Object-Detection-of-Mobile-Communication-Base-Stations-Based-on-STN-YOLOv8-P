import torch
from ultralytics import YOLO
from thop import profile

def show_flops(model_path, weights_path=None, title=""):
    model = YOLO(model_path)
    print(f"🔍 {title} Summary:")
    model.info(verbose=True)

    if weights_path:
        print(f"🔄 Loading pruned weights from {weights_path}")
        model.model.load_state_dict(torch.load(weights_path), strict=False)

    input_sample = torch.randn(1, 3, 640, 640).to(model.device)
    model.model.to(model.device).eval()
    flops, params = profile(model.model, inputs=(input_sample,))
    print(f"🔢 GFLOPs: {flops / 1e9:.2f} | Params: {params / 1e6:.2f} M\n")


# 剪枝后的模型（注意，这里加载的是剪枝后的 state_dict 权重）
show_flops('best.pt', weights_path='yolov8n_structured_pruned_weights.pt', title="Pruned YOLOv8")
